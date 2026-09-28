"""
Step A of raw-feature extraction (run with an env that has PyAV + OpenCV, e.g. envs/RoboTwin):
decode every MOSI clip straight from Raw.zip into
  <out>/audio/<vid>__<clip>.npy   float32 mono 16 kHz waveform
  <out>/faces/<vid>__<clip>.npy   uint8 (T, 224, 224, 3) face crops sampled at --fps
Utterance ids come from the MMSA pickle so the result lines up with text / labels.

usage: python decode_raw.py --pkl MOSI/unaligned_50.pkl --zip Raw.zip --out MOSI_raw_decoded [--fps 10] [--workers 12]
"""

import argparse
import io
import os
import pickle
import zipfile
from multiprocessing import Pool

import av
import cv2
import numpy as np

SR = 16000
FACE = 224
_ZIP = None
_CASCADE = None


def _init(zip_path):
    global _ZIP, _CASCADE
    _ZIP = zipfile.ZipFile(zip_path)
    _CASCADE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))


def _audio(buf):
    c = av.open(io.BytesIO(buf))
    r = av.AudioResampler(format='flt', layout='mono', rate=SR)
    xs = []
    for fr in c.decode(audio=0):
        xs += [o.to_ndarray().reshape(-1) for o in r.resample(fr)]
    xs += [o.to_ndarray().reshape(-1) for o in r.resample(None)]
    c.close()
    return np.concatenate(xs).astype(np.float32) if xs else np.zeros(SR // 10, np.float32)


def _crop(img, box):
    h, w = img.shape[:2]
    x, y, bw, bh = box
    cx, cy, s = x + bw / 2, y + bh / 2, max(bw, bh) * 1.3
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    x1, y1 = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
    return cv2.resize(img[y0:y1, x0:x1], (FACE, FACE), interpolation=cv2.INTER_AREA)


def _faces(buf, fps):
    c = av.open(io.BytesIO(buf))
    stream = c.streams.video[0]
    step, next_t = 1.0 / fps, 0.0
    out, last_box, n_det = [], None, 0
    for fr in c.decode(video=0):
        t = float(fr.pts * stream.time_base) if fr.pts is not None else next_t
        if t + 1e-6 < next_t:
            continue
        next_t += step
        img = fr.to_ndarray(format='rgb24')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        det = _CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(det):
            last_box = max(det, key=lambda b: b[2] * b[3])
            n_det += 1
        if last_box is None:                          # no face yet: centre square crop
            h, w = img.shape[:2]
            s = min(h, w)
            box = ((w - s) // 2, (h - s) // 2, s / 1.3, s / 1.3)
        else:
            box = last_box
        out.append(_crop(img, box))
    c.close()
    arr = np.stack(out) if out else np.zeros((1, FACE, FACE, 3), np.uint8)
    return arr, n_det


def work(job):
    uid, member, out, fps = job
    name = uid.replace('$_$', '__')
    pa, pf = os.path.join(out, 'audio', name + '.npy'), os.path.join(out, 'faces', name + '.npy')
    if os.path.exists(pa) and os.path.exists(pf):
        return uid, 'skip', 0, 0
    try:
        buf = _ZIP.read(member)
        a = _audio(buf)
        f, n_det = _faces(buf, fps)
        np.save(pa, a)
        np.save(pf, f)
        return uid, 'ok', len(f), n_det
    except Exception as e:  # keep going, report at the end
        return uid, f'error: {e}', 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--zip', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0)
    opt = ap.parse_args()

    data = pickle.load(open(opt.pkl, 'rb'))
    ids = [str(i) for s in ('train', 'valid', 'test') for i in data[s]['id']]
    members = set(zipfile.ZipFile(opt.zip).namelist())
    os.makedirs(os.path.join(opt.out, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(opt.out, 'faces'), exist_ok=True)

    jobs, missing = [], []
    for uid in ids:
        vid, clip = uid.split('$_$')
        m = f'Raw/{vid}/{clip}.mp4'
        (jobs.append((uid, m, opt.out, opt.fps)) if m in members else missing.append(uid))
    if opt.limit:
        jobs = jobs[:opt.limit]
    print(f'{len(jobs)} clips to decode, {len(missing)} missing in zip: {missing[:5]}', flush=True)

    n_frames, n_det, errors = 0, 0, []
    with Pool(opt.workers, initializer=_init, initargs=(opt.zip,)) as pool:
        for k, (uid, st, nf, nd) in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            n_frames += nf
            n_det += nd
            if st.startswith('error'):
                errors.append((uid, st))
            if (k + 1) % 200 == 0:
                print(f'{k+1}/{len(jobs)} done, face-detect rate {n_det/max(1,n_frames):.3f}', flush=True)
    print(f'DONE {len(jobs)} clips, {len(errors)} errors {errors[:5]}, face-detect rate {n_det/max(1,n_frames):.3f}',
          flush=True)


if __name__ == '__main__':
    main()
