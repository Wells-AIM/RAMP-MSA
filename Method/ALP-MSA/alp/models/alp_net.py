"""
ALP-Net: ALMT backbone (language-guided hyper-modality learning + cross
transformer fusion) whose per-modality token compression is replaced by a
configurable tokenizer (see patching.py).  With all tokenizers set to `almt`
this module reproduces the original ALMT model.
"""

import torch
from torch import nn
from einops import repeat
from transformers import BertModel

from .layers import Transformer, CrossTransformer, HhyperLearningEncoder
from .patching import AffectivePatcher


class BertTextEncoder(nn.Module):
    def __init__(self, pretrained, finetune=True):
        super().__init__()
        self.model = BertModel.from_pretrained(pretrained)
        self.finetune = finetune

    def forward(self, text):
        # text: (B, 3, L) = input_ids, attention_mask, token_type_ids
        ids, att, seg = text[:, 0].long(), text[:, 1].float(), text[:, 2].long()
        with torch.set_grad_enabled(self.finetune and self.training):
            return self.model(input_ids=ids, attention_mask=att, token_type_ids=seg)[0]


class ALPNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        K, D = m.token_len, m.token_dim
        self.K = K
        self.h_hyper = nn.Parameter(torch.ones(1, K, D))
        self.bert = BertTextEncoder(m.bert_pretrained, finetune=True)

        def patcher(mode, in_dim, max_len):
            return AffectivePatcher(mode, in_dim, D, K, max_len, depth=m.proj_depth, heads=m.proj_heads,
                                    mlp_dim=m.proj_mlp_dim, patch_depth=m.patch_depth, tau=m.patch_tau,
                                    b_min=m.boundary_min, use_raw_cue=m.use_raw_cue)

        assert m.text_mode != 'frame', 'text tokens act as AHL queries and must be compressed to K tokens'
        self.tok_l = patcher(m.text_mode, m.l_input_dim, m.l_input_length)
        self.tok_a = patcher(m.audio_mode, m.a_input_dim, m.a_input_length)
        self.tok_v = patcher(m.vision_mode, m.v_input_dim, m.v_input_length)

        self.l_encoder = Transformer(num_frames=K, save_hidden=True, token_len=None, dim=D,
                                     depth=m.AHL_depth - 1, heads=m.l_enc_heads, mlp_dim=m.l_enc_mlp_dim)
        self.h_hyper_layer = HhyperLearningEncoder(dim=D, depth=m.AHL_depth, heads=m.ahl_heads,
                                                   dim_head=m.ahl_dim_head, dropout=m.ahl_droup)
        self.fusion_layer = CrossTransformer(source_num_frames=K, tgt_num_frames=K, dim=D,
                                             depth=m.fusion_layer_depth, heads=m.fusion_heads,
                                             mlp_dim=m.fusion_mlp_dim)
        self.regression_layer = nn.Linear(D, 1)

    def forward(self, vision, audio, text, vision_mask, audio_mask, return_aux=False):
        b = vision.size(0)
        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=b)
        x_text = self.bert(text)
        text_mask = text[:, 1].bool()

        h_l, _, aux_l = self.tok_l(x_text, text_mask)
        h_a, mask_a, aux_a = self.tok_a(audio, audio_mask)
        h_v, mask_v, aux_v = self.tok_v(vision, vision_mask)

        h_t_list = self.l_encoder(h_l)
        h_hyper = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper, mask_a=mask_a, mask_v=mask_v)
        feat = self.fusion_layer(h_hyper, h_t_list[-1])[:, 0]
        out = self.regression_layer(feat)
        if return_aux:
            return out, {'l': aux_l, 'a': aux_a, 'v': aux_v}
        return out
