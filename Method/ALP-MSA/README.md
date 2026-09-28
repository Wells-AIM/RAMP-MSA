# ALP-MSA: Affective Latent Patching for Multimodal Sentiment Analysis

Direction 2 of *ICLR2026 emotion recognition method summary*:

> Dimension alignment ≠ semantic-granularity alignment. Text, audio and video carry
> affective information at different temporal granularities, so continuous A/V
> streams should be re-organised into **affective-event-level latent patches**
> before cross-modal fusion.

## Starting point

Backbone: **ALMT** (Zhang et al., EMNLP 2023, <https://github.com/Haoyu-ha/ALMT>, MIT).
ALMT works on the *unaligned* MMSA features (MOSI audio 375×5, vision 500×20) and
compresses every modality into K=8 learnable tokens — i.e. the "fixed latent token"
baseline that Direction 2 wants to replace.

Why not CaReFlow (CVPR 2026)? It only uses word-aligned features, where A/V are
already averaged per word (the granularity question disappears), and the official
code reproduced on this server at Acc2 84.6±2.5 / Acc7 44.7 (5 seeds, val-selected,
`RAMP-MSA/experiment_versions/careflow_official_protocol_20260923`) vs 89.8 / 50.6 reported.
Its rectified-flow fusion operates on pooled vectors, so it can later be stacked on
top of ALP tokens if useful.

## Tokenizers (RQ1: frame vs fixed vs dynamic vs event patches)

`model.{text,audio,vision}_mode` ∈

| mode | description |
|---|---|
| `almt` | original ALMT (learnable tokens, **no padding mask** — ~90% of MOSI A/V frames are padding) |
| `query` | learnable tokens with padding mask |
| `frame` | no compression, masked frame-level keys in the hyper-modality layer |
| `uniform` | K fixed-duration patches over the valid part of the sequence |
| `dynamic` | **ALP**: event-boundary scores (H-Net style adjacent-frame dissimilarity + raw prosody/expression change cue) redistribute frames over K patches; high-change regions → short patches, stationary regions → merged. Constant boundaries ⇒ exactly `uniform`. |

## Protocol

Model selection on the validation split only (`base.key_eval`, default MAE); the test
metrics reported are those of the selected epoch. (The ALMT repo reports the best
test epoch per metric, which is optimistic.) Multiple seeds, mean ± std.

## Usage

```bash
conda activate /media/disk3/muxy/envs/HME
python train.py --config configs/mosi.yaml --seed 1111 --name mosi_almt
python train.py --config configs/mosi.yaml --seed 1111 --name mosi_alp_av \
    --set model.audio_mode=dynamic model.vision_mode=dynamic
python tools/summarize.py runs
```

Data: `/media/disk3/muxy/Dataset/MSA_unaligned/{MOSI,MOSEI}/unaligned_50.pkl`
(from hf-mirror `tamb2203579/CMU-MOSI`, `tamb2203579/CMU-MOSEI`, MMSA format).
