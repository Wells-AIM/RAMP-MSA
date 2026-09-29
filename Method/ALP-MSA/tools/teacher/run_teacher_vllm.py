"""
vLLM version of run_teacher.py (same tasks / prompts / jsonl format), for throughput.
Qwen2.5-Omni thinker (text output only); EmotionThinker is a Qwen2.5-Omni-7B fine-tune.

usage (envs/vllm): python run_teacher_vllm.py --model /path/EmotionThinker --task whole --pkl ... \
       --decoded ... --out teacher_whole.jsonl --gpu 0 [--shard i --nshards n]
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_teacher import SYSTEM, TASK_PROMPTS, parse   # noqa: E402  (prompts shared with the HF script)


def build_prompt(task, text):
    mm = ''
    if task in ('whole', 'facial'):
        mm += '<|vision_bos|><|VIDEO|><|vision_eos|>'
    if task in ('whole', 'prosody'):
        mm += '<|audio_bos|><|AUDIO|><|audio_eos|>'
    q = TASK_PROMPTS[task].format(text=text.strip())
    return (f'<|im_start|>system\n{SYSTEM}<|im_end|>\n'
            f'<|im_start|>user\n{mm}{q}<|im_end|>\n<|im_start|>assistant\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--task', required=True, choices=list(TASK_PROMPTS))
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--decoded', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', default='0', help='GPU id, or comma list for tensor parallelism')
    ap.add_argument('--tp', type=int, default=1, help='tensor-parallel size (e.g. 2 for 72B-AWQ on 2x4090)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--fps', type=float, default=2.0)
    ap.add_argument('--max_frames', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=320)
    ap.add_argument('--chunk', type=int, default=64)          # prompts submitted per generate() call
    ap.add_argument('--gpu_util', type=float, default=0.9)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--splits', default='train,valid,test')
    opt = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    from vllm import LLM, SamplingParams
    use_video, use_audio = opt.task in ('whole', 'facial'), opt.task in ('whole', 'prosody')
    arch = json.load(open(os.path.join(opt.model, 'config.json'))).get('architectures', [''])[0]
    omni = 'Omni' in arch
    if not omni and (use_video or use_audio):
        raise SystemExit(f'{arch} is text-only; task {opt.task} needs audio/video')
    kw = dict(max_num_batched_tokens=16384, mm_processor_cache_gb=0,   # cache hits drop use_audio_in_video (0.10.2)
              limit_mm_per_prompt={'audio': 1, 'video': 1, 'image': 1}) if omni else {}   # a 0 disables the encoder cache
    llm = LLM(model=opt.model, max_model_len=8192 if omni else 4096, max_num_seqs=32 if omni else 64,
              gpu_memory_utilization=opt.gpu_util, seed=0, tensor_parallel_size=opt.tp, **kw)
    sp = SamplingParams(temperature=0.0, max_tokens=opt.max_new_tokens)

    data = pickle.load(open(opt.pkl, 'rb'))
    items = [(s, str(i), str(t)) for s in opt.splits.split(',')
             for i, t in zip(data[s]['id'], data[s]['raw_text'])][opt.shard::opt.nshards]
    if opt.limit:
        items = items[:opt.limit]
    done = {json.loads(l)['id'] for l in open(opt.out)} if os.path.exists(opt.out) else set()
    todo = [it for it in items if it[1] not in done]
    print(f'{len(todo)} to do ({len(done)} done)', flush=True)
    step = max(1, int(round(10 / opt.fps)))

    t0, n = time.time(), 0
    with open(opt.out, 'a') as fo:
        for s in range(0, len(todo), opt.chunk):
            chunk = todo[s:s + opt.chunk]
            reqs = []
            for split, uid, text in chunk:
                name = uid.replace('$_$', '__')
                mm = {}
                if use_audio:
                    mm['audio'] = (np.load(os.path.join(opt.decoded, 'audio', name + '.npy')), 16000)
                if use_video:
                    f = np.load(os.path.join(opt.decoded, 'faces', name + '.npy'))[::step][:opt.max_frames]
                    if len(f) % 2:
                        f = np.concatenate([f, f[-1:]])
                    mm['video'] = f
                req = {'prompt': build_prompt(opt.task, text)}
                if mm:
                    req['multi_modal_data'] = mm
                    if use_video:
                        req['mm_processor_kwargs'] = {'use_audio_in_video': False}
                reqs.append(req)
            outs = llm.generate(reqs, sp, use_tqdm=False)
            for (split, uid, _), o in zip(chunk, outs):
                raw = o.outputs[0].text
                fields, score = parse(raw)
                fo.write(json.dumps({'id': uid, 'split': split, 'task': opt.task, 'raw': raw, 'fields': fields,
                                     'score': score}, ensure_ascii=False) + '\n')
            fo.flush()
            n += len(chunk)
            print(f'{n}/{len(todo)} done, {(time.time()-t0)/n:.2f}s/item', flush=True)
    print('DONE', n, flush=True)


if __name__ == '__main__':
    main()
