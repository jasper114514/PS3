import os
from os.path import join as j_
import pdb
import torch.nn.functional as F

import numpy as np
import torch


try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print('scikit-survival not installed. Exiting...')
    raise

from src.mil_models.tokenizer import PrototypeTokenizer
from src.mil_models import create_multimodal_survival_model, prepare_emb
from src.utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss
from src.utils.utils import (EarlyStopping, save_checkpoint, AverageMeter, safe_list_to,
                             get_optim, print_network, get_lr_scheduler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROTO_MODELS = ['PANTHER', 'OT', 'H2T', 'ProtoCount']
import time

def evaluate_model(datasets, args):
    """
    Train for a single fold for suvival
    """

    writer_dir = args.results_dir
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    assert args.es_metric == 'loss'

    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
    elif args.loss_fn == 'rank':
        loss_fn = SurvRankingLoss()

    args.feat_dim = args.in_dim  # Patch feature dimension
    print('\nInit Model...', end=' ')

    # If prototype-based models, need to create slide-level embeddings
    if args.model_histo_type in PROTO_MODELS:
        datasets, _ = prepare_emb(datasets, args, mode='survival')

        new_in_dim = None
        for k, loader in datasets.items():
            assert loader.dataset.X is not None
            new_in_dim_curr = loader.dataset.X.shape[-1]
            if new_in_dim is None:
                new_in_dim = new_in_dim_curr
            else:
                assert new_in_dim == new_in_dim_curr

            # The original embedding is 1-D (long) feature vector
            # Reshape it to (n_proto, -1)
            tokenizer = PrototypeTokenizer(args.model_histo_type, args.out_type, args.n_proto)
            prob, mean, cov = tokenizer(loader.dataset.X)
            loader.dataset.X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)],
                                         dim=-1)

            factor = args.n_proto

        args.in_dim = new_in_dim // factor
    else:
        print(f"{args.model_histo_type} doesn't construct unsupervised slide-level embeddings!")

    ## Set the dimensionality for different inputs
    args.omic_dim = datasets['train'].dataset.omics_data.shape[1]

    if args.omics_modality in ['pathway', 'functional']:
        omic_sizes = datasets['train'].dataset.omic_sizes
    else:
        omic_sizes = []

    model = create_multimodal_survival_model(args, omic_sizes=omic_sizes)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model=model, args=args)
    lr_scheduler = get_lr_scheduler(args, optimizer, datasets['train'])

    if args.early_stopping:
        print('\nSetup EarlyStopping...', end=' ')
        early_stopper = EarlyStopping(save_dir=args.results_dir,
                                      patience=args.es_patience,
                                      min_stop_epoch=args.es_min_epochs,
                                      better='min' if args.es_metric == 'loss' else 'max',
                                      verbose=True)
    else:
        print('\nNo EarlyStopping...', end=' ')
        early_stopper = None

    #####################
    # The training loop #
    #####################
    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint_path)  # Ensure args.checkpoint_path is the correct .pth file

    # Load the state dictionary into the model
    model.load_state_dict(checkpoint)

    # If you are using CUDA, ensure compatibility
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print('Model Loaded')

    ### End of epoch: Evaluate on val and test set
    results, dumps = {}, {}
    for k, loader in datasets.items():
        if k.upper() == 'TRAIN':  # Skip training set
            continue
        print(f'Evaluating on Split {k.upper()}...:')
        if args.model_mm_type in ['coattn', 'coattn_text']:
            return_attn = False
        else:
            return_attn = False  # True for MMP
        results[k], dumps[k] = evaluate_survival(model, loader, loss_fn, print_every=args.print_every,
                                                 dump_results=True, return_attn=return_attn, verbose=False,
                                                 process_text=args.process_text)
        #if k == 'train':
        #    _ = results.pop('train')

    return results, dumps


def ensure_consistent_dimensions(attn_list):
    for i, attn in enumerate(attn_list):
        if attn.ndim == 2:
            attn_list[i] = np.expand_dims(attn, axis=0)
    return attn_list


@torch.no_grad()
def evaluate_survival(model, loader,
                      loss_fn=None,
                      print_every=50,
                      dump_results=False,
                      recompute_loss_at_end=True,
                      return_attn=False,
                      verbose=1, process_text=False):
    model.eval()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    all_omic_attn, all_cross_attn, all_path_attn, cross_attn_histology_text, cross_attn_pathways_text,cross_attn_text_histology,cross_attn_text_pathways  = [], [], [], [], [], [], [],
    total_time = 0
    total_samples = 0
    for batch_idx, batch in enumerate(loader):
        data = batch['img'].to(device)
        label = batch['label'].to(device)
        omics = safe_list_to(batch['omics'], device)

        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None
        case_ids = batch['case_id']

        text = safe_list_to(batch['text'], device)

        # ---------------- TIMING START ----------------
        torch.cuda.synchronize()  # <-- important if using GPU
        start = time.time()

        if process_text:
           out, log_dict = model(data, omics, text, attn_mask=attn_mask, label=label, censorship=censorship,
                                 loss_fn=loss_fn, return_attn=return_attn)
        else:
           out, log_dict = model(data, omics, attn_mask=attn_mask, label=label, censorship=censorship, loss_fn=loss_fn,
                                 return_attn=return_attn)

        # from torch.profiler import profile, record_function, ProfilerActivity
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("model_inference"):
        #         if process_text:
        #             model(data, omics, text, attn_mask=attn_mask, label=label, censorship=censorship,
        #                                                           loss_fn=loss_fn,return_attn=return_attn)
        #         else:
        #             model(data, omics, attn_mask=attn_mask, label=label, censorship=censorship, loss_fn=loss_fn,
        #               return_attn=return_attn)
        # print(prof.key_averages().table(sort_by="cuda_time_total"))

        torch.cuda.synchronize()  # <-- wait for GPU ops to finish
        end = time.time()
        # ---------------- TIMING END ----------------
        total_time += (end - start)
        total_samples += len(data)

        if return_attn:
            all_omic_attn.append(out['omic_attn'].detach().cpu().numpy())
            all_cross_attn.append(out['cross_attn'].detach().cpu().numpy())
            all_path_attn.append(out['path_attn'].detach().cpu().numpy())

            #extra
            # cross_attn_histology_text.append(out['cross_attn_histology_text'].detach().cpu().numpy())
            # cross_attn_pathways_text.append(out['cross_attn_pathways_text'].detach().cpu().numpy())
            # cross_attn_text_histology.append(out['cross_attn_text_histology'].detach().cpu().numpy())
            # cross_attn_text_pathways.append(out['cross_attn_text_pathways'].detach().cpu().numpy())

        # End of iteration survival-specific metrics to calculate / log
        bag_size_meter.update(data.size(1), n=len(data))
        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))
        all_risk_scores.append(out['risk'].cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        if verbose and (((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1)):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    if return_attn:
        if len(all_omic_attn[0].shape) == 2:
            all_omic_attn = np.stack(all_omic_attn)
            all_cross_attn = np.stack(all_cross_attn)
            all_path_attn = np.stack(all_path_attn)

            # cross_attn_histology_text = np.stack(cross_attn_histology_text)
            # cross_attn_pathways_text = np.stack(cross_attn_pathways_text)
            # cross_attn_text_histology = np.stack(cross_attn_text_histology)
            # cross_attn_text_pathways = np.stack(cross_attn_text_pathways)
        else:
            # Fix dimensions for each list
            all_omic_attn = ensure_consistent_dimensions(all_omic_attn)
            all_cross_attn = ensure_consistent_dimensions(all_cross_attn)
            all_path_attn = ensure_consistent_dimensions(all_path_attn)

            # cross_attn_histology_text = ensure_consistent_dimensions(cross_attn_histology_text)
            # cross_attn_pathways_text = ensure_consistent_dimensions(cross_attn_pathways_text)
            # cross_attn_text_histology = ensure_consistent_dimensions(cross_attn_text_histology)
            # cross_attn_text_pathways = ensure_consistent_dimensions(cross_attn_text_pathways)

            all_omic_attn = np.vstack(all_omic_attn)
            all_cross_attn = np.vstack(all_cross_attn)
            all_path_attn = np.vstack(all_path_attn)

            # cross_attn_histology_text = np.vstack(cross_attn_histology_text)
            # cross_attn_pathways_text = np.vstack(cross_attn_pathways_text)
            # cross_attn_text_histology = np.vstack(cross_attn_text_histology)
            # cross_attn_text_pathways = np.vstack(cross_attn_text_pathways)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})

    if recompute_loss_at_end and isinstance(loss_fn, CoxLoss):
        surv_loss_dict = loss_fn(logits=torch.tensor(all_risk_scores).unsqueeze(1),
                                 times=torch.tensor(all_event_times).unsqueeze(1),
                                 censorships=torch.tensor(all_censorships).unsqueeze(1))
        results['surv_loss'] = surv_loss_dict['loss'].item()
        results.update({k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})

    if verbose:
        msg = [f"{k}: {v:.3f}" for k, v in results.items()]
        print("\t".join(msg))

    dumps = {}
    if dump_results:
        dumps['all_risk_scores'] = all_risk_scores
        dumps['all_censorships'] = all_censorships
        dumps['all_event_times'] = all_event_times
        dumps['sample_ids'] = np.array(
            loader.dataset.idx2sample_df['sample_id'])
        if return_attn:
            dumps['all_omic_attn'] = all_omic_attn
            dumps['all_cross_attn'] = all_cross_attn
            dumps['all_path_attn'] = all_path_attn

            # dumps['cross_attn_histology_text'] = cross_attn_histology_text
            # dumps['cross_attn_pathways_text'] = cross_attn_pathways_text
            # dumps['cross_attn_text_histology'] = cross_attn_text_histology
            # dumps['cross_attn_text_pathways'] = cross_attn_text_pathways

    avg_time_per_sample = total_time / total_samples
    print(f"\n⚡ Average model inference time: {avg_time_per_sample:.6f} sec/sample")
    print(f"\n⚡ Total time: {total_time:.6f} ")

    return results, dumps