"""
Main entry point for survival downstream tasks
"""

from __future__ import print_function

import argparse
import pdb
import os
from os.path import join as j_
import sys

import numpy as np

from src.training.eval import evaluate_model
# internal imports
from src.utils.file_utils import save_pkl
from src.utils.utils import (seed_torch, array2list, merge_dict, read_splits,
                         parse_model_name, get_current_time, extract_patching_info)

from src.training.trainer import train
from src.wsi_datasets import WSIOmicsSurvivalDataset
# pytorch imports
import torch
from torch.utils.data import DataLoader

import pandas as pd
import json

from src.wsi_datasets.dataset_utils import collate_fn
from src.wsi_datasets.wsi_survival import WSIOmicsTextSurvivalDataset
os.environ["CUDA_VISIBLE_DEVICES"]="0"
PROTO_MODELS = ['PANTHER', 'OT', 'H2T', 'ProtoCount']

def build_datasets(csv_splits, model_type, batch_size=1, num_workers=2, train_kwargs={}, val_kwargs={}):
    """
    Construct dataloaders from the data splits
    """
    dataset_splits = {}
    label_bins = None
    for k in csv_splits.keys(): # ['train', 'val', 'test']
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy() if (k == 'train') else val_kwargs.copy()
        dataset_kwargs['label_bins'] = label_bins
        dataset = WSIOmicsTextSurvivalDataset(df_histo=df['histo'], df_gene=df['gene'], **dataset_kwargs)

        # If prototype methods, each WSI will have same feature bag dimension and is batchable
        # Otherwise, we need to use batch size of 1 to accommodate to different bag size for each WSI.
        # Alternatively, we can sample same number of patch features per WSI to have larger batch.
        if model_type not in PROTO_MODELS:
            batch_size = batch_size if dataset_kwargs.get('bag_size', -1) > 0 else 1

        if k == 'train':
            scaler = dataset.get_scaler()

        assert scaler is not None, "Omics scaler from train split required"
        dataset.apply_scaler(scaler)

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=dataset_kwargs['shuffle'], num_workers=num_workers, collate_fn=collate_fn)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')
        if (args.loss_fn == 'nll') and (k == 'train'):
            label_bins = dataset.get_label_bins()
    return dataset_splits


def main(args):
    if args.train_bag_size == -1:
        args.train_bag_size = args.bag_size
    if args.val_bag_size == -1:
        args.val_bag_size = args.bag_size
    if args.loss_fn != 'nll':
        args.n_label_bins = 0

    if args.model_mm_type.lower() == 'survpath':
        assert args.model_histo_type.lower() == 'mil', "To use SurvPath, the model_type needs to be mil"

    censorship_col = args.target_col.split('_')[0] + '_censorship'
    
    # Specify omics dir
    cancer_type = args.split_dir.split('/')[-1].split('_')[1]   # 'splits/survival/TCGA_BRCA_overall_survival_k=0' => 'BRCA'
    args.omics_dir = j_(args.omics_dir, args.type_of_path, cancer_type)

    train_kwargs = dict(data_source=args.data_source,
                        reports_dir=args.reports_dir,
                        survival_time_col=args.target_col,
                        censorship_col=censorship_col,
                        n_label_bins=args.n_label_bins,
                        label_bins=None,
                        bag_size=args.train_bag_size,
                        shuffle=True,
                        omics_dir=args.omics_dir,
                        omics_modality=args.omics_modality
                        )

    # use the whole bag at test time
    val_kwargs = dict(data_source=args.data_source,
                      reports_dir=args.reports_dir,
                      survival_time_col=args.target_col,
                      censorship_col=censorship_col,
                      n_label_bins=args.n_label_bins,
                      label_bins=None,
                      bag_size=args.val_bag_size,
                      shuffle=False,
                      omics_dir=args.omics_dir,
                      omics_modality=args.omics_modality
                      )

    all_results, all_dumps = {}, {}

    seed_torch(args.seed)
    csv_splits = read_splits(args)
    print('successfully read splits for: ', list(csv_splits.keys()))
    dataset_splits = build_datasets(csv_splits, 
                                    model_type=args.model_histo_type,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    train_kwargs=train_kwargs,
                                    val_kwargs=val_kwargs)

    if args.eval == True:
        fold_results, fold_dumps = evaluate_model(dataset_splits, args)
    else:
        fold_results, fold_dumps = train(dataset_splits, args)

    # Save results
    for split, split_results in fold_results.items():
        all_results[split] = merge_dict({}, split_results) if (split not in all_results.keys()) else merge_dict(all_results[split], split_results)
        save_pkl(j_(args.results_dir, f'{split}_results.pkl'), fold_dumps[split]) # saves per-split, per-fold results to pkl
    
    final_dict = {}
    for split, split_results in all_results.items():
        final_dict.update({f'{metric}_{split}': array2list(val) for metric, val in split_results.items()})
    final_df = pd.DataFrame(final_dict)
    save_name = 'summary.csv'
    final_df.to_csv(j_(args.results_dir, save_name), index=False)
    with open(j_(args.results_dir, save_name + '.json'), 'w') as f:
        f.write(json.dumps(final_dict, sort_keys=True, indent=4))
    
    dump_path = j_(args.results_dir, 'all_dumps.h5')
    save_pkl(dump_path, fold_dumps)

    return final_dict

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
### optimizer settings ###
FEATURE_ENCODER = 'PLIP'
print('Features used', FEATURE_ENCODER)
tissue_type = 'COADREAD'
split = 4

parser.add_argument('--max_epochs', type=int, default=50,
                    help='maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=0.00001,
                    help='weight decay')
parser.add_argument('--accum_steps', type=int, default=1,
                    help='grad accumulation steps')
parser.add_argument('--opt', type=str, default='adamW',
                    choices=['adamW', 'sgd', 'RAdam'])
parser.add_argument('--lr_scheduler', type=str,
                    choices=['cosine', 'linear', 'constant'], default='cosine')
parser.add_argument('--warmup_steps', type=int,
                    default=-1, help='warmup iterations')
parser.add_argument('--warmup_epochs', type=int,
                    default=1, help='warmup epochs')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eval', type=bool, default=False)
parser.add_argument('--checkpoint_path', type=str,  default=f'/user/Results/PS3/{FEATURE_ENCODER}/Exp/{tissue_type}_survival/k={split}/TCGA_{tissue_type}_overall_survival_k={split}::s_checkpoint.pth')
#

### misc ###
parser.add_argument('--print_every', default=100,
                    type=int, help='how often to print')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--num_workers', type=int, default=8)

### Earlystopper args ###
parser.add_argument('--early_stopping', type=int,
                    default=0, help='enable early stopping')
parser.add_argument('--es_min_epochs', type=int, default=3,
                    help='early stopping min epochs')
parser.add_argument('--es_patience', type=int, default=5,
                    help='early stopping min patience')
parser.add_argument('--es_metric', type=str, default='loss',
                    help='early stopping metric')

### model args ###
parser.add_argument('--model_histo_type', type=str, choices=['H2T', 'OT', 'PANTHER', 'ProtoCount', 'MIL'],
                    default='PANTHER', help='type of histology model')
parser.add_argument('--ot_eps', default=1, type=float,
                    help='Strength for entropic constraint regularization for OT')
parser.add_argument('--model_histo_config', type=str,
                    default='PANTHER_default', help="name of model config file")
parser.add_argument('--n_fc_layers', type=int, default=0)
parser.add_argument('--em_iter', type=int, default=1)
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--out_type', type=str, default='allcat')

# Multimodal args ###
parser.add_argument('--num_coattn_layers', default=1, type=int)
parser.add_argument('--model_mm_type', default='coattn_text',
                    help='Multimodal model type')
parser.add_argument('--attn_mode', default='full_all_es',
                    help='Attention mode type')
parser.add_argument('--append_prob', action='store_true', default=False)
parser.add_argument('--histo_agg', default='mean')
parser.add_argument('--omics_dir', default='./PS3-github/src/data_csvs/rna')
parser.add_argument('--omics_modality', default='pathway')
parser.add_argument('--type_of_path', default='hallmarks')

parser.add_argument('--residual', default=False)
parser.add_argument('--residual_type', default='all', choices=['all', 'path+hist'])

# Text args ###
# parser.add_argument('--process_text', type=lambda x: x.lower() == 'true', default=False,help="Enable or disable text processing (True or False)")
parser.add_argument('--process_text', default=True, help="Enable or disable text processing (True or False)")
parser.add_argument('--text_target_length', type=int)
parser.add_argument('--text_max_length', type=int)
parser.add_argument('--text_resizing_model', type=str, default='SA_sampling')

parser.add_argument('--net_indiv', default=True)
parser.add_argument('--net_text_combined', default=False)
parser.add_argument('--group_prototype', default=False)
parser.add_argument('--append_embed', type=str, default='random',
                    choices=['none', 'modality', 'proto', 'random', 'random_xavier', 'uniform'])

# Prototype related
parser.add_argument('--load_proto', action='store_true', default=False)
parser.add_argument('--proto_path', type=str)
parser.add_argument('--fix_proto', default=True)
parser.add_argument('--n_proto', type=int, default=16)

parser.add_argument('--in_dim', default=512, type=int,
                    help='dim of input features')
parser.add_argument('--bag_size', type=int, default='-1')
parser.add_argument('--train_bag_size', type=int, default='-1')
parser.add_argument('--val_bag_size', type=int, default='-1')
parser.add_argument('--loss_fn', type=str, default='cox', choices=['nll', 'cox', 'sumo', 'ipcwls', 'rank'],
                    help='which loss function to use')
parser.add_argument('--nll_alpha', type=float, default=0.5,
                    help='Balance between censored / uncensored loss')

# experiment task / label args ###
parser.add_argument('--exp_code', type=str, default=None,
                    help='experiment code for saving results')
parser.add_argument('--task', type=str, default=f'{tissue_type}_survival')
parser.add_argument('--target_col', type=str, default='dss_survival_days')
parser.add_argument('--n_label_bins', type=int, default=4,
                    help='number of bins for event time discretization')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=f'/user/Data/TCGA/{tissue_type}/{FEATURE_ENCODER}_features/feats_pt',
                    help='manually specify the data source')
parser.add_argument('--reports_dir', type=str, default=f'/user/TCGA/{tissue_type}/{FEATURE_ENCODER}_TCGA_Reports/',
                    help='manually specify the reports dir')
parser.add_argument('--split_dir', type=str, default=f'./PS3-github/PS3_Splits/tcga-{tissue_type.lower()}/TCGA_{tissue_type}_overall_survival_k={split}',
                    help='manually specify the set of splits to use')
parser.add_argument('--split_names', type=str, default='train,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--overwrite', default=True,
                    help='overwrite existing results')

# logging args ###
parser.add_argument('--results_dir', default=f'/user/PS3/Results/{tissue_type}/k={split}',
                    help='results directory (default: ./results)')
parser.add_argument('--tags', nargs='+', type=str, default=None,
                    help='tags for logging')

parser.add_argument('--wandb_project', default='mmp_final')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    print('task: ', args.task)
    args.split_dir = j_('splits', args.split_dir)
    print('split_dir: ', args.split_dir)
    split_num = args.split_dir.split('/')[7].split('_k=')
    args.split_name_clean = args.split_dir.split('/')[7]
    if len(split_num) > 1:
        args.split_k = int(split_num[1])
    else:
        args.split_k = 0

    if args.load_proto:
        assert os.path.isfile(args.proto_path), f"Path {args.proto_path} doesn't exist!"
        args.proto_fname = '/'.join(args.proto_path.split('/')[-2:])
        proto_fname_clean = '--'.join(args.proto_fname[:-4].split('/'))

    ### Allows you to pass in multiple data sources (separated by comma). If single data source, no change.
    args.data_source = [src for src in args.data_source.split(',')]
    check_params_same = []
    for src in args.data_source: 
        ### assert data source exists + extract feature name ###
        print('data source: ', src)
        assert os.path.isdir(src), f"data source must be a directory: {src} invalid"

        ### parse patching info ###
        feat_name = os.path.basename(src)


        #### parse model name ####
        parsed = parse_model_name(feat_name)
        parsed.update({'patch_mag': 20, 'patch_size': 224})

        
    ### Updated parsed mdoel parameters in args.Namespace ###
    for key, val in parsed.items():
        setattr(args, key, val)

    ### Updated text in args.Namespace ###
    csv_file = f'{args.split_dir}/train.csv'  # Replace with the path to your CSV file

    # Read the case IDs from the CSV file
    case_ids = pd.read_csv(csv_file)['case_id'].astype(str).tolist()  # Assuming column is named 'case_id'

    # Filter .pt files that match the case IDs
    pt_files = [os.path.join(args.reports_dir, f) for f in os.listdir(args.reports_dir)
                if f.endswith('.pt') and os.path.splitext(f)[0] in case_ids]

    if not pt_files:
        raise FileNotFoundError("No matching files found in the directory.")

    # Extract lengths of embeddings
    lengths = []
    for file in pt_files:
        embedding = torch.load(file)  # Load .pt file
        lengths.append(embedding.shape[0])  # Get number of chunks (first dimension)

    # Analyze length statistics
    max_length = max(lengths)
    mean_length = int(np.mean(lengths))
    median_length = int(np.median(lengths))
    percentile_90 = int(np.percentile(lengths, 90))

    text_parameters = {
            "text_max_length": max_length,
            "text_target_length": mean_length,
        }

    for key, val in text_parameters.items():
        setattr(args, key, val)

    print(f'Text Target Length : {args.text_target_length}')
    print(f'Text Max Length : {args.text_max_length}')

    ### setup results dir ### es = extra softmax
    if args.exp_code is None:
        if args.process_text:
            if args.model_mm_type == 'text_baseline':
                exp_code = f"{args.split_name_clean}::{args.model_histo_config}::{feat_name}::{args.model_mm_type}"
            else:
                exp_code = (f"{args.split_name_clean}::{args.model_histo_config}::{feat_name}::{args.model_mm_type}:"
                        f"{args.text_target_length}::net_indiv_{args.net_indiv}::append_embed_{args.append_embed}::attn_mode_{args.attn_mode}")
        else:
            exp_code = f"{args.split_name_clean}::{args.model_histo_config}::{feat_name}::{args.model_mm_type}"
    else:
        pass


    args.results_dir = j_(args.results_dir, 
                          args.task, 
                          f'k={args.split_k}', 
                          str(exp_code))

    os.makedirs(args.results_dir, exist_ok=True)


    print("\n################### Settings ###################")
    for key, val in vars(args).items():
        print("{}:  {}".format(key, val))

    with open(j_(args.results_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    #### train ####
    results = main(args)

    print("FINISHED!\n\n\n")