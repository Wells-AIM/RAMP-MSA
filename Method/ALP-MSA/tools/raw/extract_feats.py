"""
Step B of raw-feature extraction (HME env, GPU): turn decoded clips into frame-level features
and write an npy directory readable by alp/data.py (same keys as tools/prepare_npy.py).

  audio  : WavLM-base+ hidden states, mean of layers [--wavlm_layers], 50 Hz, 768-d
  vision : CLIP ViT-B/32 image embeddings of face crops, --fps (from step A), 512-d
  text_bert / labels / split / ids: copied from the MMSA pickle (unchanged text pipeline)

usage: python extract_feats.py --pkl MOSI/unaligned_50.pkl --decoded MOSI_raw_decoded --out MOSI_wavlm_clip/npy
"""

import argparse
import json
import os
import pickle

import numpy as np
import torch
from transformers import CLIPVisionModelWithProjection, WavLMModel

EXP = '/media/disk3/muxy/Method/RAMP-MSA/experiment_versions'
WAVLM = f'{EXP}/wavlm_audio_20260920/models/wavlm-base-plus'
CLIP = f'{EXP}/clip_vision_20260921/models/clip-vit-base-patch32'
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_wavlm(path, device):
    """WavLMModel.from_pretrained with transformers 4.30 + torch>=2.1 silently re-initialises the
    weight-normed positional conv (checkpoint: weight_g/weight_v, model: parametrizations.*).
    Copy those two tensors explicitly and verify."""
    model = WavLMModel.from_pretrained(path)
    sd = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location='cpu')
    conv = model.encoder.pos_conv_embed.conv
    g, v = sd['encoder.pos_conv_embed.conv.weight_g'], sd['encoder.pos_conv_embed.conv.weight_v']
    if hasattr(conv, 'parametrizations'):
        conv.parametrizations.weight.original0.data.copy_(g)
        conv.parametrizations.weight.original1.data.copy_(v)
        assert torch.equal(conv.parametrizations.weight.original1.data, v)
    else:
        conv.weight_g.data.copy_(g)
        conv.weight_v.data.copy_(v)
    return model.to(device).eval()


@torch.no_grad()
def wavlm_feats(model, wav, layers, device, chunk_s=20):
    wav = (wav - wav.mean()) / (wav.std() + 1e-7)                   # WavLM feature-extractor normalisation
    outs, step = [], 16000 * chunk_s
    for s in range(0, len(wav), step):
        x = torch.from_numpy(wav[s:s + step]).float()[None].to(device)
        if x.shape[1] < 400:                                         # shorter than one conv receptive field
            x = torch.nn.functional.pad(x, (0, 400 - x.shape[1]))
        hs = model(x, output_hidden_states=True).hidden_states       # tuple(13) of (1, T, 768)
        outs.append(torch.stack([hs[i] for i in layers]).mean(0)[0].float().cpu())
    return torch.cat(outs).numpy()


@torch.no_grad()
def clip_feats(model, faces, device, bs=128):
    outs = []
    for s in range(0, len(faces), bs):
        x = torch.from_numpy(faces[s:s + bs]).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - CLIP_MEAN) / CLIP_STD).to(device)
        outs.append(model(pixel_values=x).image_embeds.float().cpu())
    return torch.cat(outs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--decoded', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--wavlm_layers', default='7,8,9,10,11,12')
    ap.add_argument('--max_audio', type=int, default=1000)   # 20 s at 50 Hz
    ap.add_argument('--max_vision', type=int, default=300)   # 30 s at 10 fps
    ap.add_argument('--gpu', type=int, default=0)
    opt = ap.parse_args()
    device = torch.device(f'cuda:{opt.gpu}')
    layers = [int(x) for x in opt.wavlm_layers.split(',')]

    wavlm = load_wavlm(WAVLM, device)
    clip = CLIPVisionModelWithProjection.from_pretrained(CLIP).to(device).eval()
    wavlm_dtype = torch.float16  # autocast; weights stay fp32 (LayerNorm etc. in fp32)

    data = pickle.load(open(opt.pkl, 'rb'))
    os.makedirs(opt.out, exist_ok=True)
    meta, stats = {}, {}
    for split in ('train', 'valid', 'test'):
        d = data[split]
        ids = [str(i) for i in d['id']]
        N = len(ids)
        A = np.lib.format.open_memmap(os.path.join(opt.out, f'{split}_audio.npy'), 'w+', np.float16,
                                      (N, opt.max_audio, 768))
        V = np.lib.format.open_memmap(os.path.join(opt.out, f'{split}_vision.npy'), 'w+', np.float16,
                                      (N, opt.max_vision, 512))
        alen, vlen = np.zeros(N, np.int64), np.zeros(N, np.int64)
        for i, uid in enumerate(ids):
            name = uid.replace('$_$', '__')
            wav = np.load(os.path.join(opt.decoded, 'audio', name + '.npy'))
            faces = np.load(os.path.join(opt.decoded, 'faces', name + '.npy'))
            with torch.autocast('cuda', dtype=wavlm_dtype):
                fa = wavlm_feats(wavlm, wav, layers, device)
                fv = clip_feats(clip, faces, device)
            la, lv = min(len(fa), opt.max_audio), min(len(fv), opt.max_vision)
            A[i, :la], V[i, :lv] = fa[:la], fv[:lv]
            A[i, la:], V[i, lv:] = 0, 0
            alen[i], vlen[i] = max(la, 1), max(lv, 1)
            if (i + 1) % 200 == 0:
                print(f'{split} {i+1}/{N}', flush=True)
        A.flush(); V.flush()
        np.save(os.path.join(opt.out, f'{split}_text_bert.npy'), np.asarray(d['text_bert'], np.float32))
        np.save(os.path.join(opt.out, f'{split}_regression_labels.npy'), np.asarray(d['regression_labels'], np.float32))
        np.save(os.path.join(opt.out, f'{split}_audio_lengths.npy'), alen)
        np.save(os.path.join(opt.out, f'{split}_vision_lengths.npy'), vlen)
        json.dump(ids, open(os.path.join(opt.out, f'{split}_id.json'), 'w'))
        meta[split] = {'text_bert': [N, 3, 50], 'audio': [N, opt.max_audio, 768], 'vision': [N, opt.max_vision, 512],
                       'regression_labels': [N], 'audio_lengths': [N], 'vision_lengths': [N]}
        stats[split] = {'audio_len_median': float(np.median(alen)), 'audio_len_max': int(alen.max()),
                        'vision_len_median': float(np.median(vlen)), 'vision_len_max': int(vlen.max()),
                        'audio_truncated': int((alen >= opt.max_audio).sum()),
                        'vision_truncated': int((vlen >= opt.max_vision).sum())}
        print(split, stats[split], flush=True)
    json.dump(meta, open(os.path.join(opt.out, 'meta.json'), 'w'), indent=1)
    json.dump({'wavlm_layers': layers, 'audio_hz': 50, 'vision_fps': 'see decode step', 'stats': stats},
              open(os.path.join(opt.out, 'features_info.json'), 'w'), indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
