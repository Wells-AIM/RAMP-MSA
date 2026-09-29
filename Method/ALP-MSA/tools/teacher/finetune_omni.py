"""
Strong teacher for Direction 3: Qwen2.5-Omni-7B thinker, 4-bit NF4 + LoRA, discriminative regression readout
(final-layer hidden state of the last prompt token -> linear head), following "Discriminative Hidden-State
Readout from a Native Omni-Modal LLM for MSA" (arXiv 2606.05713).

K-fold CROSS-FITTING on the training split (folds grouped by video = speaker): the model of fold k is trained on
the other folds (model selection on the official valid split) and predicts fold k, valid and test.
Merged with merge_folds.py, every TRAIN utterance gets a teacher output from a model that never saw it.

usage (envs/omni): python finetune_omni.py --fold 0 --nfolds 5 --mode text --gpu 0 --out_dir .../omni_ft
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

SYSTEM = ('You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving '
          'auditory and visual inputs, as well as generating text and speech.')
QUESTION = ('Transcript: "{text}"\nHow positive or negative is the sentiment the speaker expresses in this '
            'utterance, on a scale from -3 (highly negative) to +3 (highly positive)?')
LORA_TARGET = r'.*model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)'   # LLM only


def fold_of(uid, nfolds):
    vid = uid.split('$_$')[0]
    return int(hashlib.md5(vid.encode()).hexdigest(), 16) % nfolds


class Teacher(torch.nn.Module):
    def __init__(self, path, lora_r, text_only=False):
        super().__init__()
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 llm_int8_skip_modules=['audio_tower', 'visual', 'lm_head'])
        full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            path, quantization_config=bnb, torch_dtype=torch.bfloat16, enable_audio_output=False,
            attn_implementation='sdpa', device_map={'': 0})
        thinker = full.thinker
        if text_only:   # encoders are never called for text-only prompts: keep them off the GPU
            thinker.audio_tower.to('cpu')
            thinker.visual.to('cpu')
            torch.cuda.empty_cache()
        # (peft's prepare_model_for_kbit_training upcasts every non-quantised weight to fp32 -> OOM on 24 GB)
        for p in thinker.parameters():
            p.requires_grad_(False)
        thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        thinker.enable_input_require_grads()
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_r, lora_dropout=0.1, target_modules=LORA_TARGET,
                         bias='none')
        self.thinker = get_peft_model(thinker, cfg)
        hidden = full.thinker.config.text_config.hidden_size
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, inputs):
        out = self.thinker(**inputs, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][:, -1].float()          # batch 1 / left padding: last token = last prompt token
        return self.head(h).squeeze(-1), h


def build_inputs(processor, mode, text, wav, faces, fps):
    content = []
    if 'video' in mode or mode == 'all':
        content.append({'type': 'video'})
    if 'audio' in mode or mode == 'all':
        content.append({'type': 'audio'})
    content.append({'type': 'text', 'text': QUESTION.format(text=text.strip())})
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': content}]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    prompt = prompt[0] if isinstance(prompt, list) else prompt
    kw = {}
    if faces is not None:
        kw['videos'] = [faces]
        kw['fps'] = fps
    if wav is not None:
        kw['audio'] = [wav]
    return processor(text=prompt, return_tensors='pt', padding=True, use_audio_in_video=False, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/media/disk3/muxy/Dataset/pretrained_models/Qwen2.5-Omni-7B')
    ap.add_argument('--pkl', default='/media/disk3/muxy/Dataset/MSA_unaligned/MOSI/unaligned_50.pkl')
    ap.add_argument('--decoded', default='/media/disk3/muxy/Dataset/MSA_raw_feats/MOSI/decoded')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--mode', default='text', choices=['text', 'text_video', 'text_audio', 'all'])
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--nfolds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--head_lr', type=float, default=1e-3)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--lora_r', type=int, default=32)
    ap.add_argument('--fps', type=float, default=2.0)
    ap.add_argument('--max_frames', type=int, default=16)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='debug: cap every split')
    opt = ap.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(opt.gpu))
    random.seed(opt.seed); np.random.seed(opt.seed); torch.manual_seed(opt.seed)
    device = torch.device('cuda:0')
    os.makedirs(opt.out_dir, exist_ok=True)
    tag = f'{opt.mode}_fold{opt.fold}of{opt.nfolds}'

    data = pickle.load(open(opt.pkl, 'rb'))
    items = {s: [(str(i), str(t), float(y)) for i, t, y in
                 zip(data[s]['id'], data[s]['raw_text'], data[s]['regression_labels'])] for s in data}
    train = [it for it in items['train'] if fold_of(it[0], opt.nfolds) != opt.fold]
    held = [it for it in items['train'] if fold_of(it[0], opt.nfolds) == opt.fold]
    valid, test = items['valid'], items['test']
    if opt.limit:
        train, held, valid, test = train[:opt.limit], held[:opt.limit], valid[:opt.limit], test[:opt.limit]
    print(f'{tag}: train {len(train)} held-out {len(held)} valid {len(valid)} test {len(test)}', flush=True)

    processor = Qwen2_5OmniProcessor.from_pretrained(opt.model)
    model = Teacher(opt.model, opt.lora_r, text_only=(opt.mode == 'text'))
    model.head.to(device)
    step = max(1, int(round(10 / opt.fps)))

    def inputs_for(uid, text):
        name = uid.replace('$_$', '__')
        wav = faces = None
        if 'audio' in opt.mode or opt.mode == 'all':
            wav = np.load(os.path.join(opt.decoded, 'audio', name + '.npy'))
        if 'video' in opt.mode or opt.mode == 'all':
            faces = np.load(os.path.join(opt.decoded, 'faces', name + '.npy'))[::step][:opt.max_frames]
            if len(faces) % 2:
                faces = np.concatenate([faces, faces[-1:]])
        x = build_inputs(processor, opt.mode, text, wav, faces, opt.fps)
        return {k: (v.to(device, torch.bfloat16) if v.is_floating_point() else v.to(device)) for k, v in x.items()}

    @torch.no_grad()
    def predict(split_items):
        model.eval()
        preds, hids = [], []
        for uid, text, _ in split_items:
            p, h = model(inputs_for(uid, text))
            preds.append(float(p))
            hids.append(h[0].cpu().numpy().astype(np.float16))
        model.train()
        return np.array(preds, np.float32), np.stack(hids)

    lora_params = [p for n, p in model.thinker.named_parameters() if p.requires_grad]
    optim = torch.optim.AdamW([{'params': lora_params, 'lr': opt.lr},
                               {'params': model.head.parameters(), 'lr': opt.head_lr}], weight_decay=0.01)
    total = opt.epochs * math.ceil(len(train) / opt.accum)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    best, best_ep, t0 = None, -1, time.time()
    model.train()
    for ep in range(1, opt.epochs + 1):
        order = list(range(len(train)))
        random.shuffle(order)
        run = 0.0
        for k, i in enumerate(order):
            uid, text, y = train[i]
            p, _ = model(inputs_for(uid, text))
            loss = torch.nn.functional.mse_loss(p, torch.tensor([y], device=device)) / opt.accum
            loss.backward()
            run += float(loss) * opt.accum
            if (k + 1) % opt.accum == 0 or k + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(lora_params + list(model.head.parameters()), 1.0)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            if (k + 1) % 200 == 0:
                print(f'ep {ep} {k+1}/{len(order)} loss {run/(k+1):.4f} {time.time()-t0:.0f}s', flush=True)
        vp, _ = predict(valid)
        vy = np.array([y for _, _, y in valid])
        mae = float(np.mean(np.abs(vp - vy)))
        print(f'ep {ep} valid MAE {mae:.4f} corr {np.corrcoef(vp, vy)[0,1]:.4f} {time.time()-t0:.0f}s', flush=True)
        if best is None or mae < best:
            best, best_ep = mae, ep
            model.thinker.save_pretrained(os.path.join(opt.out_dir, f'{tag}_lora'))
            torch.save(model.head.state_dict(), os.path.join(opt.out_dir, f'{tag}_head.pt'))

    if best_ep != opt.epochs:   # reload the val-selected epoch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(os.path.join(opt.out_dir, f'{tag}_lora', 'adapter_model.safetensors'))
        set_peft_model_state_dict(model.thinker, sd)
        model.head.load_state_dict(torch.load(os.path.join(opt.out_dir, f'{tag}_head.pt')))
    res = {}
    for name, split_items in (('held', held), ('valid', valid), ('test', test)):
        p, h = predict(split_items)
        res[f'{name}_ids'] = np.array([u for u, _, _ in split_items])
        res[f'{name}_pred'], res[f'{name}_hid'] = p, h
        y = np.array([yy for _, _, yy in split_items])
        nz = y != 0
        print(f'{name}: MAE {np.mean(np.abs(p - y)):.4f} corr {np.corrcoef(p, y)[0,1]:.4f} '
              f'acc2(non0) {np.mean((p[nz] > 0) == (y[nz] > 0)):.4f}', flush=True)
    np.savez(os.path.join(opt.out_dir, f'{tag}.npz'), **res)
    json.dump({'fold': opt.fold, 'nfolds': opt.nfolds, 'mode': opt.mode, 'best_epoch': best_ep, 'best_valid_mae': best,
               'time_s': round(time.time() - t0)}, open(os.path.join(opt.out_dir, f'{tag}.json'), 'w'))
    print('DONE', tag, flush=True)


if __name__ == '__main__':
    main()
