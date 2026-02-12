import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.utils import safe_list_to
from .components import SNN_Block, FeedForward, FeedForwardEnsemble, process_surv, Attn_Net_Gated, \
    PS3AttentionLayer, FeedForwardEnsemble_combined_text
from .text_processing import SelfAttentionResizer, interpolate_to_fixed_length


def init_per_path_model(omic_sizes, hidden_dim=256):
    """
    Create a list of SNNs, one for each pathway

    Args:
        omic_sizes: List of integers, each indicating number of genes per prototype
    """
    hidden = [hidden_dim, hidden_dim]
    sig_networks = []
    for input_dim in omic_sizes:
        fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
        sig_networks.append(nn.Sequential(*fc_omic))
    sig_networks = nn.ModuleList(sig_networks)

    return sig_networks

def agg_histo(X, agg_mode='mean'):
    """
    Aggregating histology
    """
    if agg_mode == 'mean':
        out = torch.mean(X, dim=1)
    elif agg_mode == 'cat':
        out = X.reshape(X.shape[0], -1)
    else:
        raise NotImplementedError(f"Not implemented for {agg_mode}")

    return out



def construct_proto_embedding(path_proj_dim, append_embed='modality', numOfproto_histo=16, numOfproto_omics=50):
    """
    Per-prototype learnable/non-learnable embeddings to append to the original prototype embeddings 
    """
    if append_embed == 'modality':  # One-hot encoding for two modalities
        path_proj_dim_new = path_proj_dim + 2

        histo_embedding = torch.tensor([[[1, 0]]]).repeat(1, numOfproto_histo, 1)  # (1, numOfproto, 2)
        gene_embedding = torch.tensor([[[0, 1]]]).repeat(1, numOfproto_omics, 1)  # (1, len(omic_sizes),2 )

    elif append_embed == 'proto':
        path_proj_dim_new = path_proj_dim + numOfproto_histo + numOfproto_omics
        embedding = torch.eye(numOfproto_histo + numOfproto_omics).unsqueeze(0)

        histo_embedding = embedding[:, :numOfproto_histo, :]  # (1, numOfproto, numOftotalproto)
        gene_embedding = embedding[:, numOfproto_histo:, :]  # (1, len(omic_sizes), numOftotalproto)

    elif append_embed == 'random':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim

        histo_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_histo, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_omics, append_dim), requires_grad=True)

    else:
        path_proj_dim_new = path_proj_dim
        histo_embedding = None
        gene_embedding = None

    return path_proj_dim_new, histo_embedding, gene_embedding

def construct_proto_embedding_text(path_proj_dim, append_embed='modality', numOfproto_histo=16, numOfproto_omics=50, numOfproto_text=43):
    """
    Per-prototype learnable/non-learnable embeddings to append to the original prototype embeddings
    """
    if append_embed == 'modality':  # One-hot encoding for two modalities
        path_proj_dim_new = path_proj_dim + 3  # Add 3 dimensions for three modalities

        histo_embedding = torch.tensor([[[1, 0, 0]]]).repeat(1, numOfproto_histo,
                                                             1)  # Histology (1, numOfproto_histo, 3)
        gene_embedding = torch.tensor([[[0, 1, 0]]]).repeat(1, numOfproto_omics, 1)  # Omics (1, numOfproto_omics, 3)
        text_embedding = torch.tensor([[[0, 0, 1]]]).repeat(1, numOfproto_text, 1)  # New modality

    elif append_embed == 'proto':
        path_proj_dim_new = path_proj_dim + numOfproto_histo + numOfproto_omics + numOfproto_text
        embedding = torch.eye(numOfproto_histo + numOfproto_omics+ numOfproto_text).unsqueeze(0)

        histo_embedding = embedding[:, :numOfproto_histo, :]  # (1, numOfproto_histo, numOftotalproto)
        gene_embedding = embedding[:, numOfproto_histo:numOfproto_histo + numOfproto_omics, :]  # (1, numOfproto_omics, numOftotalproto)
        text_embedding = embedding[:, numOfproto_histo + numOfproto_omics:, :]  # (1, numOfproto_new_modality, numOftotalproto)

    elif append_embed == 'random':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim

        histo_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_histo, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_omics, append_dim), requires_grad=True)
        text_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_text, append_dim), requires_grad=True)

    elif append_embed == 'random_xavier':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim
        histo_embedding = torch.nn.Parameter(torch.empty(1, numOfproto_histo, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.empty(1, numOfproto_omics, append_dim), requires_grad=True)
        text_embedding = torch.nn.Parameter(torch.empty(1, numOfproto_text, append_dim), requires_grad=True)
        nn.init.xavier_uniform_(histo_embedding)
        nn.init.xavier_uniform_(gene_embedding)
        nn.init.xavier_uniform_(text_embedding)

    elif append_embed == 'uniform':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim
        histo_embedding = torch.nn.Parameter(torch.empty(1, numOfproto_histo, append_dim).uniform_(-0.1, 0.1), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.empty(1, numOfproto_omics, append_dim).uniform_(-0.1, 0.1),requires_grad=True)
        text_embedding = torch.nn.Parameter(torch.empty(1, numOfproto_text, append_dim).uniform_(-0.1, 0.1),requires_grad=True)

    else:
        path_proj_dim_new = path_proj_dim
        histo_embedding = None
        gene_embedding = None
        text_embedding = None

    return path_proj_dim_new, histo_embedding, gene_embedding, text_embedding


def construct_proto_embedding_grouped_text(path_proj_dim, append_embed='modality', numOfproto_histo=16, numOfproto_omics=50, numOfproto_text=1, text_batch_size=20):
    """
    Per-prototype learnable/non-learnable embeddings to append to the original prototype embeddings
    """
    if append_embed == 'modality':  # One-hot encoding for two modalities
        path_proj_dim_new = path_proj_dim + 3  # Add 3 dimensions for three modalities

        histo_embedding = torch.tensor([[[1, 0, 0]]]).repeat(1, numOfproto_histo, 1)  # Histology (1, numOfproto_histo, 3)
        gene_embedding = torch.tensor([[[0, 1, 0]]]).repeat(1, numOfproto_omics, 1)  # Omics (1, numOfproto_omics, 3)
        text_embedding = torch.tensor([[[0, 0, 1]]]).repeat(1, text_batch_size, 1)  # New modality

    elif append_embed == 'proto':
        path_proj_dim_new = path_proj_dim + numOfproto_histo + numOfproto_omics + numOfproto_text
        embedding = torch.eye(numOfproto_histo + numOfproto_omics + numOfproto_text).unsqueeze(0)

        histo_embedding = embedding[:, :numOfproto_histo, :]  # (1, numOfproto_histo, numOftotalproto)
        gene_embedding = embedding[:, numOfproto_histo:numOfproto_histo + numOfproto_omics, :]  # (1, numOfproto_omics, numOftotalproto)
        text_embedding = embedding[:, numOfproto_histo + numOfproto_omics:, :].repeat(1, text_batch_size, 1)

    elif append_embed == 'random':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim

        histo_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_histo, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_omics, append_dim), requires_grad=True)
        text_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_text, append_dim), requires_grad=True).repeat(1, text_batch_size, 1)

    else:
        path_proj_dim_new = path_proj_dim
        histo_embedding = None
        gene_embedding = None
        text_embedding = None

    return path_proj_dim_new, histo_embedding, gene_embedding, text_embedding


################################
# Multimodal fusion approaches #
################################

class coattn_text(nn.Module):
    def __init__(
            self,
            omic_sizes=[100, 200, 300, 400, 500, 600],
            histo_in_dim=1024,
            dropout=0.1,
            num_classes=4,
            path_proj_dim=256,
            num_coattn_layers=1,
            modality='both',
            histo_agg='mean',
            histo_model='PANTHER',
            append_embed='none',
            group_prototype = False,
            mult=1,
            net_indiv=False,
            net_text_combined = False,
            numOfproto=16,
            text_target_length = 43,
            text_max_length = 60,
            text_resizing_model = None,
            attn_mode = None,
            residual = False,
            residual_type = 'all'
    ):
        """
        The central co-attention module where you can do it all!

        Args:
            omic_sizes: List of integers, each indicating number of genes per prototype
            histo_in_dim: Dimension of histology feature embedding
            num_classes: 4 if we are using NLL, 1 if we are using Cox/Ranking loss
            path_proj_dim: Dimension of the embedding space where histology and pathways are fused
            modality: ['gene','histo','coattn', 'partial'] 'coattn' accounts for both modalities
                If 'histo' or 'gene', unimodal self-attention
            histo_agg: ['mean', 'cat'] Take average of post-attention embeddings ('mean') or concatenate ('cat')
            histo_model: ['mil','PANTHER', 'OT', 'H2T']: 'mil' is for non-prototype-based methods
            net_indiv (bool): If True, create FFN for each prototype
            numOfproto: Number of histology prototypes
        """

        super().__init__()

        self.num_pathways = len(omic_sizes)
        self.num_coattn_layers = num_coattn_layers

        self.histo_in_dim = histo_in_dim
        self.out_mult = mult
        self.net_indiv = net_indiv
        self.net_text_combined = net_text_combined
        self.group_prototype = group_prototype
        self.modality = modality

        self.histo_agg = histo_agg

        self.numOfproto = numOfproto
        self.num_classes = num_classes

        self.histo_model = histo_model.lower()

        self.sig_networks = init_per_path_model(omic_sizes)
        self.identity = nn.Identity()  # use this layer to calculate ig

        self.append_embed = append_embed

        self.text_target_length = text_target_length
        self.max_text_length = text_max_length
        self.text_resizing_model = text_resizing_model
        self.text_in_dim = self.histo_in_dim

        if self.text_resizing_model == 'SA_sampling':
            self.text_resizer = SelfAttentionResizer(input_dim=self.text_in_dim, max_length=self.max_text_length, target_length=self.text_target_length, aggregation_method='sampling')
            self.text_resize_dim = 512

        if self.histo_model == 'panther':  # Uses prob/mean/cov
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim * 2 + 1, path_proj_dim))
        else:
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim, path_proj_dim))

        self.text_proj_net = nn.Sequential(nn.Linear(self.text_resize_dim, path_proj_dim))

        if self.histo_model != "mil":
            if self.group_prototype:
                self.path_proj_dim, self.histo_embedding, self.gene_embedding, self.text_embedding = (construct_proto_embedding_grouped_text
                                                                                                      (path_proj_dim,
                                                                                                      self.append_embed,
                                                                                                      self.numOfproto,
                                                                                                      len(omic_sizes),
                                                                                                       1,
                                                                                                       text_target_length))
            else:
                self.path_proj_dim, self.histo_embedding, self.gene_embedding, self.text_embedding = construct_proto_embedding_text(path_proj_dim,
                                                                                                      self.append_embed,
                                                                                                      self.numOfproto,
                                                                                                      len(omic_sizes),
                                                                                                      text_target_length)
        else:
            self.path_proj_dim = path_proj_dim
            self.histo_embedding = None
            self.gene_embedding = None
            self.text_embedding = None

        coattn_list = []
        if self.num_coattn_layers == 0:
            out_dim = self.path_proj_dim

            if self.net_indiv:  # Individual MLP per prototype
                feed_forward = FeedForwardEnsemble(out_dim,
                                                   self.out_mult,
                                                   dropout=dropout,
                                                   num=self.numOfproto + len(omic_sizes)+ self.text_target_length)
            else:
                feed_forward = FeedForward(out_dim, self.out_mult, dropout=dropout)

            layer_norm = nn.LayerNorm(int(out_dim * self.out_mult))
            coattn_list.extend([feed_forward, layer_norm])
        else:
            out_dim = self.path_proj_dim // 2
            out_mult = self.out_mult

            if self.modality in ['histo', 'gene', 'text']:  # If we want to use only single modality + self-attention
                attn_mode = 'self'
            else:
                attn_mode = attn_mode  # Otherwise, perform self & cross attention
            print('Attn Mode: ', attn_mode)
            cross_attender = PS3AttentionLayer(
                dim=self.path_proj_dim,
                dim_head=out_dim,
                heads=1,
                residual=residual,
                dropout=0.1,
                num_pathways=self.num_pathways,
                num_prototypes=self.numOfproto,
                attn_mode=attn_mode,
                residual_type='all'
            )

            if self.net_indiv:
                if self.net_text_combined:
                    feed_forward = FeedForwardEnsemble_combined_text(out_dim, # Individual MLP per prototype but combined for text
                                                   out_mult,
                                                   dropout=dropout,
                                                   num=self.numOfproto + len(omic_sizes))
                else:
                    feed_forward = FeedForwardEnsemble(out_dim, # Individual MLP per prototype and for text token
                                                   out_mult,
                                                   dropout=dropout,
                                                   num=self.numOfproto + len(omic_sizes) + self.text_target_length)
            else:
                feed_forward = FeedForward(out_dim, out_mult, dropout=dropout)

            layer_norm = nn.LayerNorm(int(out_dim * out_mult))
            coattn_list.extend([cross_attender, feed_forward, layer_norm])

        self.coattn = nn.Sequential(*coattn_list)

        out_dim_final = int(out_dim * self.out_mult)
        histo_final_dim = out_dim_final * self.numOfproto if self.histo_agg == 'cat' else out_dim_final
        gene_final_dim = out_dim_final
        text_final_dim = out_dim_final

        if self.modality == 'histo':
            in_dim = histo_final_dim
        elif self.modality == 'gene':
            in_dim = gene_final_dim
        elif self.modality == 'text':
            in_dim = text_final_dim
        elif self.modality == 'histo+text':
            in_dim = histo_final_dim + text_final_dim
        elif self.modality == 'pathways+text':
            in_dim = gene_final_dim + text_final_dim
        else:
            in_dim = histo_final_dim + gene_final_dim + text_final_dim

        self.classifier = nn.Linear(in_dim, self.num_classes, bias=False)

    def forward_no_loss(self, x_path, x_omics, x_text, return_attn=False):
        """
        Args:
            x_path: (B, numOfproto, in_dim) in_dim = [prob, mean, cov] (If OT, prob will be uniform, cov will be none)
            x_omics:
            return_attn:

        """
        device = x_path.device

        ## Pathway embeddings
        h_omic = []  ## each omic signature goes through it's own FC layer
        for idx, sig_feat in enumerate(x_omics):
            omic_feat = self.sig_networks[idx](sig_feat.float())  # (B, d)
            h_omic.append(omic_feat)
        h_omic = torch.stack(h_omic, dim=1)  # [batch_size, 50, out_feats = 256]

        if self.gene_embedding is not None:  # Append gene prototype encoding
            arr = []
            for idx in range(len(h_omic)):
                arr.append(torch.cat([h_omic[idx:idx + 1], self.gene_embedding.to(device)], dim=-1))
            h_omic = torch.cat(arr, dim=0)  # [batch_size, 50, out_feats + prototype encoding = 288 ]

        ## Histology embeddings
        # Project wsi to smaller dimension (same as pathway dimension)
        h_path = self.path_proj_net(x_path)  # [batch_size, 16, out_feats = 256 ]

        if self.histo_embedding is not None:  # Append histo prototype encoding
            arr = []
            for idx in range(len(h_path)):
                arr.append(torch.cat([h_path[idx:idx + 1], self.histo_embedding.to(device)], dim=-1))
            h_path = torch.cat(arr, dim=0)  # [batch_size, 50, out_feats + prototype encoding = 288]

        if self.text_resizing_model == 'Interpolation':
            h_text = interpolate_to_fixed_length(x_text, self.text_target_length) # [batch_size, text_target_length, out_feats = 512]
        else:
            h_text = self.text_resizer(x_text, device)  # [batch_size, text_target_length, out_feats = 768]

        h_text = safe_list_to(h_text, device)

        h_text = self.text_proj_net(h_text) # [batch_size, text_target_length, out_feats = 256]

        if self.text_embedding is not None:
            arr = []
            for idx in range(len(h_text)):
                arr.append(torch.cat([h_text[idx:idx + 1], self.text_embedding.to(device)], dim=-1)) # [batch_size, text_length, out_feats + prototype encoding = 288]
            h_text = torch.cat(arr, dim=0)  # [batch_size, text_target_length, out_feats + prototype encoding = 288]

        tokens = torch.cat([h_omic, h_path, h_text], dim=1)  # (B, N_p+N_h, d)
        tokens = self.identity(tokens)  # []

        # Required for visualization
        if return_attn:
            with torch.no_grad():
                #_, attn_pathways, cross_attn_pathways, cross_attn_histology = self.coattn[0](x=tokens, mask=None,return_attention=True)
                _, attn_pathways, cross_attn_pathways, cross_attn_histology, cross_attn_histology_text, cross_attn_pathways_text, cross_attn_text_histology, cross_attn_text_pathways = \
                self.coattn[0](x=tokens, mask=None, return_attention=True)

        # Pass the token set through co-attention network
        mm_embed = self.coattn(tokens)

        # ---> aggregate
        # Pathways
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        # Histology
        start_histology = self.num_pathways
        end_histology = self.num_pathways + self.numOfproto
        wsi_postSA_embed = mm_embed[:, start_histology:end_histology, :]

        # Text
        text_postSA_embed = mm_embed[:, end_histology:, :]  # Remaining tokens are for text
        text_postSA_embed = torch.mean(text_postSA_embed, dim=1)

        if self.histo_model == 'mil':
            wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)  # For non-prototypes, we just take the mean
        else:
            wsi_postSA_embed = agg_histo(wsi_postSA_embed, self.histo_agg)

        if self.modality == 'histo':  # Just use histo for prediction
            embedding = wsi_postSA_embed
        elif self.modality == 'gene':  # Just use gene for prediction
            embedding = paths_postSA_embed
        elif self.modality == 'text':  # Use only text
            embedding = text_postSA_embed
        else:  # Use both modalities
            embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed, text_postSA_embed], dim=1)  # ---> both branches

        logits = self.classifier(embedding)
        out = {'logits': logits}
        if return_attn:
            out['omic_attn'] = attn_pathways
            out['cross_attn'] = cross_attn_pathways
            out['path_attn'] = cross_attn_histology
            #extra
            out['cross_attn_histology_text'] = cross_attn_histology_text
            out['cross_attn_pathways_text'] = cross_attn_pathways_text
            out['cross_attn_text_histology'] = cross_attn_text_histology
            out['cross_attn_text_pathways'] = cross_attn_text_pathways

        return out

    def forward(self, x_path, x_omics, x_text, return_attn=False, attn_mask=None, label=None, censorship=None, loss_fn=None):

        out = self.forward_no_loss(x_path, x_omics, x_text, return_attn)
        results_dict, log_dict = process_surv(out['logits'], label, censorship, loss_fn)
        if return_attn:
            results_dict['omic_attn'] = out['omic_attn']
            results_dict['cross_attn'] = out['cross_attn']
            results_dict['path_attn'] = out['path_attn']
            #extra
            results_dict['cross_attn_histology_text'] = out['cross_attn_histology_text']
            results_dict['cross_attn_pathways_text'] = out['cross_attn_pathways_text']
            results_dict['cross_attn_text_histology'] = out['cross_attn_text_histology']
            results_dict['cross_attn_text_pathways'] = out['cross_attn_text_pathways']


        results_dict.update(out)
        return results_dict, log_dict

