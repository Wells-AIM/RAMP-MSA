"""
Direction 3 teacher: structured multimodal emotion-evidence reasoning with an Omni speech/vision LLM
(EmotionThinker = Qwen2.5-Omni-7B fine-tuned for prosody-aware emotion reasoning, or the base Omni).

Inputs per utterance: face frames (from tools/raw/decode_raw.py, sub-sampled to --fps), 16 kHz audio,
and the transcript (MMSA raw_text).  Output (jsonl, one line per utterance):
  {"id", "split", "raw", "fields": {semantic, prosody, facial, consistency, conclusion}, "score"}

The teacher never sees labels.  Only train-split outputs are used by the student, and only as a
training signal; inference of the student uses no teacher.

usage (env with transformers==4.52.3, qwen-omni-utils):
  python run_teacher.py --model /path/EmotionThinker --pkl MOSI/unaligned_50.pkl \
      --decoded MSA_raw_feats/MOSI/decoded --out teacher_et.jsonl --gpu 0 --shard 0 --nshards 3
"""

import argparse
import json
import os
import pickle
import re
import time

import numpy as np
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

SYSTEM = ("You are a helpful assistant. The user asks a question, and you solve it. You first think about "
          "the reasoning process in the mind and then provide the user with the answer. The reasoning "
          "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.")

PROMPT = """You are given a short clip of a person speaking: sampled face frames (video), the speech audio, and the transcript.
Transcript: "{text}"

Judge the speaker's sentiment using evidence from all three modalities. Inside <think>, write exactly these five lines:
Semantic: what the words themselves convey about sentiment.
Prosody: pitch, energy, speaking rate, pauses and voice quality, and what they convey.
Facial: facial expressions and head movements, and what they convey.
Consistency: whether the three modalities agree, and which one is most reliable here.
Conclusion: the overall sentiment and its strength.
Inside <answer>, give only one sentiment intensity score from -3.0 (highly negative) to 3.0 (highly positive) with one decimal."""

FIELDS = ('semantic', 'prosody', 'facial', 'consistency', 'conclusion')

# Modality-specific evidence prompts (one teacher pass per task). The whole-chain task sees all
# three modalities; the evidence tasks see ONLY their own modality so each field is modality-pure.
TASK_PROMPTS = {
    'whole': PROMPT,
    'prosody': ("Listen only to HOW the speaker talks, not to what the words mean. In <think>, describe the "
                "pitch, energy, speaking rate, pauses, stress and voice quality, and what emotion and sentiment "
                "they convey. In <answer>, give one sentiment intensity score from -3.0 to 3.0."),
    'facial': ("These are face frames of a person speaking (no audio). In <think>, describe the facial "
               "expressions, eye and mouth movements, head movements and their changes over time, and what "
               "emotion and sentiment they convey. In <answer>, give one sentiment intensity score from -3.0 to 3.0."),
    'semantic': ('Here is only the transcript of an utterance: "{text}". In <think>, explain what sentiment the '
                 'words themselves convey (word choice, negation, intensifiers, context). In <answer>, give one '
                 'sentiment intensity score from -3.0 to 3.0.'),
}


def parse(raw):
    think = re.search(r'<think>(.*?)(</think>|$)', raw, re.S)
    body = think.group(1) if think else raw
    fields = {}
    for i, f in enumerate(FIELDS):
        nxt = '|'.join(FIELDS[i + 1:]) or 'zzzz'
        m = re.search(rf'{f}\s*:\s*(.*?)(?=\n\s*(?:{nxt})\s*:|$)', body, re.S | re.I)
        fields[f] = m.group(1).strip() if m else ''
    ans = re.search(r'<answer>(.*?)</answer>', raw, re.S)
    num = re.search(r'[-+]?\d+(?:\.\d+)?', ans.group(1) if ans else raw.split('</think>')[-1])
    score = float(np.clip(float(num.group()), -3, 3)) if num else None
    return fields, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--decoded', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--fps', type=float, default=2.0)       # frames fed to the teacher (decoded at 10 fps)
    ap.add_argument('--max_frames', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=384)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--splits', default='train,valid,test')
    ap.add_argument('--task', default='whole', choices=list(TASK_PROMPTS))
    ap.add_argument('--batch', type=int, default=4)
    opt = ap.parse_args()

    device = f'cuda:{opt.gpu}'
    processor = Qwen2_5OmniProcessor.from_pretrained(opt.model)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        opt.model, torch_dtype=torch.bfloat16, device_map=device, attn_implementation='sdpa')
    if hasattr(model, 'disable_talker'):
        model.disable_talker()
    model.eval()

    data = pickle.load(open(opt.pkl, 'rb'))
    items = [(s, str(i), str(t)) for s in opt.splits.split(',')
             for i, t in zip(data[s]['id'], data[s]['raw_text'])]
    items = items[opt.shard::opt.nshards]
    if opt.limit:
        items = items[:opt.limit]
    done = set()
    if os.path.exists(opt.out):
        done = {json.loads(l)['id'] for l in open(opt.out)}
    print(f'{len(items)} items in shard, {len(done)} already done', flush=True)

    step = max(1, int(round(10 / opt.fps)))
    processor.tokenizer.padding_side = 'left'            # batched generation
    use_video, use_audio = opt.task in ('whole', 'facial'), opt.task in ('whole', 'prosody')
    todo = [it for it in items if it[1] not in done]
    if use_video:   # batch clips of similar length together (minimal frame padding)
        def n_frames(uid):
            f = np.load(os.path.join(opt.decoded, 'faces', uid.replace('$_$', '__') + '.npy'), mmap_mode='r')
            return min(opt.max_frames, (len(f) + step - 1) // step)
        todo.sort(key=lambda it: n_frames(it[1]))
    t0, n = time.time(), 0
    with open(opt.out, 'a') as fo:
        for s in range(0, len(todo), opt.batch):
            chunk = todo[s:s + opt.batch]
            prompts, wavs, vids = [], [], []
            for split, uid, text in chunk:
                name = uid.replace('$_$', '__')
                content = []
                if use_video:
                    faces = np.load(os.path.join(opt.decoded, 'faces', name + '.npy'))[::step][:opt.max_frames]
                    if len(faces) % 2:                  # Qwen2.5-VL temporal patch = 2 frames
                        faces = np.concatenate([faces, faces[-1:]])
                    vids.append(faces)
                    content.append({'type': 'video'})
                if use_audio:
                    wavs.append(np.load(os.path.join(opt.decoded, 'audio', name + '.npy')))
                    content.append({'type': 'audio'})
                content.append({'type': 'text', 'text': TASK_PROMPTS[opt.task].format(text=text.strip())})
                conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
                        {'role': 'user', 'content': content}]
                p = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                prompts.append(p[0] if isinstance(p, list) else p)   # 4.52 returns a list for one conv
            if vids:                                    # equal frame counts within a batch (repeat last frame)
                T = max(len(f) for f in vids)
                vids = [np.concatenate([f, np.repeat(f[-1:], T - len(f), 0)]) if len(f) < T else f for f in vids]
            inputs = processor(text=prompts, audio=wavs or None, videos=vids or None, return_tensors='pt',
                               padding=True, use_audio_in_video=False, fps=opt.fps)
            inputs = inputs.to(device).to(model.dtype)
            with torch.no_grad():
                out = model.generate(**inputs, use_audio_in_video=False, return_audio=False,
                                     max_new_tokens=opt.max_new_tokens, do_sample=False)
            raws = processor.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            for (split, uid, _), raw in zip(chunk, raws):
                fields, score = parse(raw)
                fo.write(json.dumps({'id': uid, 'split': split, 'task': opt.task, 'raw': raw, 'fields': fields,
                                     'score': score}, ensure_ascii=False) + '\n')
            fo.flush()
            n += len(chunk)
            if (s // opt.batch) % 10 == 0:
                print(f'{n}/{len(todo)} done, {(time.time()-t0)/n:.2f}s/item', flush=True)
    print('DONE', n, flush=True)


if __name__ == '__main__':
    main()
