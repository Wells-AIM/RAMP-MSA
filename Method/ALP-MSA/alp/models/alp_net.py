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

from .layers import Transformer, CrossTransformer, HhyperLearningEncoder, HhyperLearningLayer

HhyperLearningLayerScale = HhyperLearningLayer.GAMMA_SCALE
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
                                    b_min=m.boundary_min, use_raw_cue=m.use_raw_cue,
                                    frame_pos_std=getattr(m, "frame_pos_std", 0.02))

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

        # ALMT's prediction only sees the hyper tokens (text acts solely as attention queries over
        # A/V values; the text tokens fed to the fusion layer never reach the output token).
        # text_residual adds a direct text path: y = W [LN(fused); LN(mean text tokens)].
        self.text_residual = getattr(m, 'text_residual', False)
        self.text_residual_drop = getattr(m, 'text_residual_drop', 0.0)
        if self.text_residual:
            self.norm_feat, self.norm_text = nn.LayerNorm(D), nn.LayerNorm(D)
            self.regression_layer = nn.Linear(2 * D, 1)

        if getattr(m, 'av_gate', False):
            for layer in self.h_hyper_layer.layers:
                layer.fn.use_gate = True

        # event-synchronous fusion (relative-time bias between text and A/V tokens)
        self.time_bias = getattr(m, 'time_bias', False)
        if self.time_bias:
            timed = ('uniform', 'dynamic', 'uniq', 'dynq')
            assert m.text_mode in timed, 'time_bias needs timed text tokens (text_mode uniform|dynamic)'
            assert m.audio_mode in timed + ('frame',) and m.vision_mode in timed + ('frame',)
            init = m.time_bias_init / HhyperLearningLayerScale
            for layer in self.h_hyper_layer.layers:
                layer.fn.gamma_a.data.fill_(init)
                layer.fn.gamma_v.data.fill_(init)

    @staticmethod
    def _token_times(aux, mask):
        """Relative temporal centre in [0, 1] of every token: patch centres or frame positions."""
        if 'desc' in aux:
            return aux['desc'][..., 1]
        n = mask.sum(1, keepdim=True).clamp(min=1).float()
        return torch.arange(mask.size(1), device=mask.device)[None].float() / n

    def forward(self, vision, audio, text, vision_mask, audio_mask, return_aux=False):
        b = vision.size(0)
        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=b)
        x_text = self.bert(text)
        text_mask = text[:, 1].bool()

        h_l, _, aux_l = self.tok_l(x_text, text_mask)
        h_a, mask_a, aux_a = self.tok_a(audio, audio_mask)
        h_v, mask_v, aux_v = self.tok_v(vision, vision_mask)

        times = None
        if self.time_bias:
            times = (self._token_times(aux_l, text_mask), self._token_times(aux_a, audio_mask),
                     self._token_times(aux_v, vision_mask))

        h_t_list = self.l_encoder(h_l)
        h_hyper = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper, mask_a=mask_a, mask_v=mask_v, times=times)
        feat = self.fusion_layer(h_hyper, h_t_list[-1])[:, 0]
        if self.text_residual:
            t_feat = self.norm_text(h_t_list[-1].mean(1))
            if self.training and self.text_residual_drop > 0:
                # anti-shortcut: per-sample dropout of the direct text path, so the A/V-grounded
                # hyper-token path must carry the prediction on its own
                keep = (torch.rand(b, 1, device=t_feat.device) >= self.text_residual_drop).to(t_feat.dtype)
                t_feat = t_feat * keep
            feat = torch.cat([self.norm_feat(feat), t_feat], -1)
        out = self.regression_layer(feat)
        if return_aux:
            return out, {'l': aux_l, 'a': aux_a, 'v': aux_v}
        return out
