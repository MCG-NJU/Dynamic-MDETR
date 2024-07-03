import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import grid_sample

from pytorch_pretrained_bert.modeling import BertModel
from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer
from.vl_encoder import build_vl_encoder
from utils.box_utils import xywh2xyxy

import math

class DynamicMDETR(nn.Module):
    def __init__(self, args):
        super(DynamicMDETR, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len
        self.uniform_grid = args.uniform_grid
        self.uniform_learnable = args.uniform_learnable
        self.different_transformer = args.different_transformer

        self.visumodel = build_detr(args)
        self.textmodel = build_bert(args)

        num_total = self.num_visu_token + self.num_text_token
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.vl_encoder = build_vl_encoder(args)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        # Sampling relevant
        self.visual_feature_map_h = 20
        self.visual_feature_map_w = 20
        self.in_points = args.in_points
        self.stages = args.stages

        self.offset_generators = nn.ModuleList([nn.Linear(hidden_dim, self.in_points * 2) for i in range(args.stages)])
        self.update_sampling_queries = nn.ModuleList(
            [MLP(2 * hidden_dim, hidden_dim, hidden_dim, 2) for i in range(args.stages)])

        self.init_reference_point = nn.Embedding(1, 2)
        self.init_sampling_feature = nn.Embedding(1, hidden_dim)

        self.init_weights()
        if self.different_transformer:
            self.vl_transformer = nn.ModuleList([build_vl_transformer(args) for i in range(args.stages)])
        else:
            self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if self.uniform_grid:
            h = int(math.sqrt(self.in_points))
            w = h
            step = 1 / h
            start = 1 / h / 2

            new_h = torch.tensor([start + i * step for i in range(h)]).view(-1, 1).repeat(1, w)
            new_w = torch.tensor([start + j * step for j in range(w)]).repeat(h, 1)
            grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
            grid = grid.view(-1, 2)  # (in_points, 2)
            self.initial_sampled_points = torch.nn.Parameter(grid.unsqueeze(0))  # (1, in_points, 2)

    def init_weights(self):
        nn.init.constant_(self.init_reference_point.weight[:, 0], 0.5)
        nn.init.constant_(self.init_reference_point.weight[:, 1], 0.5)
        self.init_reference_point.weight.requires_grad=False

        for i in range(self.stages):
            nn.init.zeros_(self.offset_generators[i].weight)
            nn.init.uniform_(self.offset_generators[i].bias, -0.5, 0.5)
        if not self.uniform_learnable:
            self.offset_generators[0].weight.requires_grad = False
            self.offset_generators[0].bias.requires_grad = False

    def feautures_sampling(self, sampling_query, reference_point, feature_map, pos, stage):
        bs, channel = sampling_query.shape
        if self.uniform_grid:
            if stage != 0:
                xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
                sampled_points = (xy_offsets.permute(1, 0, 2) + reference_point).permute(1, 0, 2)  # (bs, in_points, 2)
            else:
                sampled_points = self.initial_sampled_points.clone().repeat(bs, 1, 1)
        else:
            xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
            sampled_points = (xy_offsets.permute(1, 0, 2) + reference_point).permute(1, 0, 2)  # (bs, in_points, 2)
        feature_map = feature_map.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w) # (bs, channel, h, w)
        pos = pos.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w) # (bs, channel, h, w)

        # [0,1] to [-1,1]
        sampled_points = (2 * sampled_points) - 1

        sampled_features = grid_sample(feature_map, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border',
                                       align_corners=False).squeeze(-1)  # (bs, channel, in_points)
        pe = grid_sample(pos, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1) # (bs, channel, in_points)

        return sampled_features, pe

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]

        # 1. Feature Encoder

        # 1.1 Visual Encoder
        # visual backbone
        out, visu_pos = self.visumodel(img_data)
        visu_mask, visu_src = out # (B, H*W), (H*W, B, channel)
        visu_src = self.visu_proj(visu_src)  # (H*W, B, channel)

        # 1.2 Language Encoder
        # language bert
        text_fea = self.textmodel(text_data)
        text_src, text_mask = text_fea.decompose()
        assert text_mask is not None
        # text_src: (bs, max_len, channel)
        text_mask = text_mask.flatten(1)  # (B, max_len)
        text_src = self.text_proj(text_src).permute(1, 0, 2)  # (max_len, B, channel)

        # 1.3 Concat visual features and language features
        vl_src = torch.cat([visu_src, text_src], dim=0)
        vl_mask = torch.cat([visu_mask, text_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)


        # 2. Multimodal Transformer
        # 2.1 Multimodal Transformer Encoder
        if self.vl_encoder is not None:
            vl_feat = self.vl_encoder(vl_src, vl_mask, vl_pos)  # (L+N)xBxC
        else:
            vl_feat = vl_src

        # 2.2 Split back to visual features and language features, use language features as queries
        visu_feat = vl_feat[:self.num_visu_token] # (H*W, B, channel)
        language_feat = vl_feat[self.num_visu_token:] # (max_len, B, channel)
        v_pos = vl_pos[:self.num_visu_token]
        l_pos = vl_pos[self.num_visu_token:]

        # 2.3 Dynamic Multimodal Transformer Decoder
        # Initialize sampling query and reference point for the first features sampling
        sampling_query = self.init_sampling_feature.weight.repeat(bs, 1)
        reference_point = self.init_reference_point.weight.repeat(bs, 1)
        pred_box = None

        for i in range(0, self.stages):
            # 2D adaptive sampling
            sampled_features, pe = self.feautures_sampling(sampling_query, reference_point, visu_feat.permute(1, 2, 0), v_pos.permute(1, 2, 0), i)

            # Text guided decoding with one-layer transformer encoder-decoder
            if self.different_transformer:
                vg_hs = self.vl_transformer[i](sampled_features, None, language_feat, pe, text_mask, l_pos)[0]
            else:
                vg_hs = self.vl_transformer(sampled_features, None, language_feat, pe, text_mask, l_pos)[0]

            # Prediction Head
            language_feat = vg_hs[0]

            text_select = (1 - text_mask * 1.0).unsqueeze(-1)  # (bs, max_len, 1)
            text_select_num = text_select.sum(dim=1)  # (bs, 1)

            # new language queries
            vg_hs = (text_select * vg_hs[0].permute(1,0,2)).sum(dim=1) / text_select_num  # (bs, channel)

            pred_box = self.bbox_embed(vg_hs).sigmoid()

            # Update reference point and sampling query
            reference_point = pred_box[:, :2]
            sampling_query = self.update_sampling_queries[i](torch.cat((vg_hs, sampling_query), dim=1))

        return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            # x = F.relu(layer(x), inplace=True) if i < self.num_layers - 1 else layer(x)
        return x


class PositionalEncodingSine(nn.Module):
    def __init__(self, emb_size: int, maxlen: int = 20):
        super(PositionalEncodingSine, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = pos_embedding

    def forward(self, token_embedding):
        return self.pos_embedding[:token_embedding.size(0), :]
