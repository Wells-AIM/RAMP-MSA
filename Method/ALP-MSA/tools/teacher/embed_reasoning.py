"""
Encode teacher reasoning (tools/teacher/run_teacher.py jsonl) into fixed vectors with a frozen
sentence encoder (CLS pooling, L2-normalised), one vector per evidence field plus the whole chain.

Output npz: ids, split, score (nan if unparsed), z_all, z_semantic, z_prosody, z_facial,
z_consistency, z_conclusion (each N x d), valid_<field> masks.

usage: python embed_reasoning.py --jsonl teacher_et.jsonl [more.jsonl ...] --encoder /path/bge-base-en-v1.5 --out teacher_et.npz
"""

import argparse
import json
import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

FIELDS = ('semantic', 'prosody', 'facial', 'consistency', 'conclusion')


@torch.no_grad()
def encode(texts, tok, model, device, bs=64):
    out = []
    for s in range(0, len(texts), bs):
        b = tok(texts[s:s + bs], padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        h = model(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h, dim=-1).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, model.config.hidden_size), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', nargs='+', required=True)
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--score_task', default='whole', help='teacher task whose score is used for label distillation')
    opt = ap.parse_args()

    # records keyed by (utterance id, task); task 'whole' gives the full chain + score, the modality-specific
    # tasks ('semantic', 'prosody', 'facial') give the evidence fields
    recs = {}
    for p in opt.jsonl:
        for line in open(p):
            r = json.loads(line)
            recs[(r['id'], r.get('task', 'whole'))] = r
    ids = sorted({i for i, _ in recs})
    device = f'cuda:{opt.gpu}'
    tok = AutoTokenizer.from_pretrained(opt.encoder)
    model = AutoModel.from_pretrained(opt.encoder).to(device).eval()

    def think(r):
        m = re.search(r'<think>(.*?)(</think>|$)', r['raw'], re.S)
        return (m.group(1) if m else r['raw']).strip()

    def field_text(i, k):
        if (i, k) in recs:                         # modality-specific teacher pass
            return think(recs[(i, k)])
        w = recs.get((i, 'whole'))
        return w['fields'].get(k, '') if w else ''

    whole_rec = [recs.get((i, 'whole')) for i in ids]
    split_of = {i: r['split'] for (i, _), r in recs.items()}
    score_rec = [recs.get((i, opt.score_task)) for i in ids]
    res = {'ids': np.array(ids), 'split': np.array([split_of[i] for i in ids]),
           'score': np.array([np.nan if (w is None or w['score'] is None) else w['score'] for w in score_rec],
                             np.float32)}
    for k in ('semantic', 'prosody', 'facial'):   # per-modality teacher scores (analysis only)
        res[f'score_{k}'] = np.array([np.nan if (i, k) not in recs or recs[(i, k)]['score'] is None
                                      else recs[(i, k)]['score'] for i in ids], np.float32)
    whole = [think(w) if w else '' for w in whole_rec]
    res['valid_all'] = np.array([bool(t) for t in whole])
    res['z_all'] = encode([t or '-' for t in whole], tok, model, device)
    for k in FIELDS:
        texts = [field_text(i, k) for i in ids]
        valid = np.array([bool(t.strip()) for t in texts])
        z = np.zeros((len(ids), res['z_all'].shape[1]), np.float32)
        if valid.any():
            z[valid] = encode([t for t, v in zip(texts, valid) if v], tok, model, device)
        res[f'z_{k}'], res[f'valid_{k}'] = z, valid
    np.savez(opt.out, **res)
    print('saved', opt.out, {k: v.shape for k, v in res.items()},
          {k: float(res[f'valid_{k}'].mean()) for k in FIELDS},
          'score parsed', float(np.isfinite(res['score']).mean()), flush=True)


if __name__ == '__main__':
    main()
