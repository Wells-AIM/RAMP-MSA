"""Map CaReFlow (word-aligned) segment ids 'vid[k]' to MMSA utterance ids 'vid$_$j'.

Matching key: (video id, normalised transcript, label rounded to 1e-3). The mapping is only used to
attach Direction-3 teacher targets to CaReFlow TRAIN samples; nothing about the test set is used.

usage: python map_ids.py --careflow datasets/mosi.pkl --mmsa .../MOSI/unaligned_50.pkl --out datasets/mosi_id_map.json
"""

import argparse
import json
import pickle
import re
from collections import defaultdict
from difflib import SequenceMatcher


def norm(t):
    t = str(t).lower().replace("'", '')            # CaReFlow transcripts drop apostrophes ("didnt")
    return re.sub(r'[^a-z0-9]+', ' ', t).strip()


def fuzzy(vid, text, label, by_vid, thr=0.8):
    """Fallback: same video, same label, unique best transcript similarity >= thr."""
    cands = [(SequenceMatcher(None, text, t).ratio(), i, s) for i, t, y, s in by_vid[vid] if abs(y - label) < 1e-3]
    cands.sort(reverse=True)
    if cands and cands[0][0] >= thr and (len(cands) == 1 or cands[1][0] < cands[0][0] - 0.05):
        return cands[0][1], cands[0][2]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--careflow', required=True)
    ap.add_argument('--mmsa', required=True)
    ap.add_argument('--out', required=True)
    opt = ap.parse_args()

    cf = pickle.load(open(opt.careflow, 'rb'))
    mm = pickle.load(open(opt.mmsa, 'rb'))
    index, by_vid = defaultdict(list), defaultdict(list)
    for s in mm:
        for i, t, y in zip(mm[s]['id'], mm[s]['raw_text'], mm[s]['regression_labels']):
            vid = str(i).split('$_$')[0]
            index[(vid, norm(t), round(float(y), 3))].append((str(i), s))
            by_vid[vid].append((str(i), norm(t), float(y), s))

    mapping, stats = {}, defaultdict(lambda: [0, 0, 0, 0])
    for split in cf:
        for (words, _, _), label, seg in cf[split]:
            vid = seg.split('[')[0]
            text, y = norm(' '.join(words)), float(label[0][0])
            hits = index.get((vid, text, round(y, 3)), [])
            st = stats[split]
            hit = hits[0] if len(hits) == 1 else None
            if hit is None:
                hit = fuzzy(vid, text, y, by_vid)
                st[3] += hit is not None
            if hit is not None:
                mapping[seg] = hit[0]
                st[0] += 1
                if {'dev': 'valid'}.get(split, split) != hit[1]:
                    st[2] += 1                       # split disagreement between the two releases
            else:
                st[1] += 1
    json.dump(mapping, open(opt.out, 'w'))
    for split, (ok, miss, cross, fz) in stats.items():
        print(f'{split:5s} matched {ok} / {ok + miss} (fuzzy {fz})  unmatched {miss}  split-mismatch {cross}')


if __name__ == '__main__':
    main()
