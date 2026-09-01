# RAMP-MSA

**RAMP: Regret-Aware Multimodal Procedural Memory for Multimodal Sentiment Analysis**

Research prototype implementing the method discussed in this conversation: instead of storing class centroids or retrieving past samples, the model stores **procedural fusion knowledge** — *how the T/A/V interactions should be corrected when a similar multimodal interaction pattern appears again*.

The package is self-contained PyTorch code and supports:

- CMU-MOSI regression (`[-3, 3]`)
- CMU-MOSEI regression (`[-3, 3]`)
- IEMOCAP classification (generic processed feature format; default config assumes 4 classes)
- Synthetic regression/classification smoke tests
- Static-memory / no-memory ablations through config overrides
- Missing/noisy modality evaluation
- Memory-gain diagnostics by interaction difficulty

> This is a **new research implementation**, not a reproduction of an already-published RAMP paper. The method is designed to be experimentally falsifiable. Do not assume SOTA before running the real benchmarks and ablations.

---

## 1. Core method

For every sample, the shared coalition fusion network is evaluated on all seven non-empty modality subsets:

`T, A, V, TA, TV, AV, TAV`.

A Möbius inversion produces seven interaction components:

- unimodal effects: `I_T, I_A, I_V`
- pairwise interactions: `I_TA, I_TV, I_AV`
- three-way interaction: `I_TAV`

These components play two roles:

1. **Memory address**: their joint pattern is compressed into an interaction-state key.
2. **Procedural action space**: a 7-D policy says which interaction components should be amplified or suppressed.

During training, an **oracle local fusion policy** is obtained from the gradient of the task loss with respect to seven temporary interaction coefficients. The external memory stores:

- `key`: when this policy is useful;
- `value`: the 7-D loss-reducing fusion policy;
- `utility`: how much paired loss reduction the policy historically produced.

At inference, no labels or gradients are required. The current interaction state retrieves a policy from memory and applies it as a residual correction to the ordinary fused representation.

Online memory consolidation uses:

- hardness: base task loss;
- novelty: distance to existing memory keys;
- stability: agreement of the oracle policy under two feature-dropout views;
- regret/utility: paired `base_loss - memory_loss` credit;
- redundancy: near-duplicate memory keys are easier to evict.

See `RESEARCH_STORY.md` for the full paper narrative.

---

## 2. Installation

```bash
cd RAMP-MSA
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested in this package with modern PyTorch. CUDA is optional.

---

## 3. Run the smoke test first

```bash
python train.py --config configs/synthetic_regression.yaml
```

Classification smoke test:

```bash
python train.py --config configs/synthetic_classification.yaml
```

The first warm-up epoch trains the ordinary parametric model and collects interaction-key / oracle-policy pairs. The memory is then clustered and initialized, after which retrieval and online consolidation are enabled.

Outputs are written under the config's `experiment.output_dir`, including:

- `best.pt`
- `last.pt`
- `resolved_config.yaml`
- `history.json/jsonl`
- `best_valid_metrics.json`
- `test_metrics.json`

---

## 4. CMU-MOSI / CMU-MOSEI

The loader directly supports the common **THUIAR/MMSA** processed pickle structure:

```python
{
    "train": {
        "text": ...,
        "audio": ...,
        "vision": ...,
        "audio_lengths": ...,
        "vision_lengths": ...,
        "regression_labels": ...,
        "id": ...,
    },
    "valid": {...},
    "test": {...},
}
```

The official MMSA repository describes this feature layout and provides `aligned_50.pkl` / `unaligned_50.pkl` files for MOSI/MOSEI.

### MOSI

```bash
python train.py \
  --config configs/mosi.yaml \
  --data-path /path/to/MOSI/Processed/unaligned_50.pkl
```

or

```bash
bash scripts/run_mosi.sh /path/to/MOSI/Processed/unaligned_50.pkl
```

### MOSEI

```bash
python train.py \
  --config configs/mosei.yaml \
  --data-path /path/to/MOSEI/Processed/unaligned_50.pkl
```

The temporal encoders uniformly subsample very long unaligned audio/vision sequences to the limits in the YAML config (`max_audio_tokens`, `max_vision_tokens`) before the Transformer. This keeps the first research version practical on a single GPU.

---

## 5. IEMOCAP

IEMOCAP preprocessing is less standardized across repositories. This package expects a split dictionary containing dense sequence features. The default config expects:

```python
{
  "train": {
    "text": [N, Lt, Dt],
    "audio": [N, La, Da],
    "vision": [N, Lv, Dv],
    "labels": [N],
    "id": [N],
  },
  "valid": {...},
  "test": {...},
}
```

Run:

```bash
python inspect_data.py /path/to/iemocap.pkl
```

Then edit `configs/iemocap.yaml -> data.keys` if your key names differ.

```bash
python train.py --config configs/iemocap.yaml --data-path /path/to/iemocap.pkl
```

Default `num_classes: 4`. Change it if your protocol uses a different class set.

---

## 6. Important evaluation metrics

For MOSI/MOSEI the package reports:

- MAE
- Pearson Corr
- Acc-7
- Acc-5
- Acc-3
- binary accuracy/F1 under both common zero-handling conventions

For IEMOCAP it reports:

- Accuracy
- Macro-F1
- Weighted-F1

Every evaluation also reports:

- base-model metrics (same network without the memory correction)
- mean paired memory gain
- mean memory gate
- low/mid/high interaction-difficulty buckets

The last item directly tests the paper hypothesis: **procedural memory should help more when modality interactions are complex.**

---

## 7. Missing/noisy modality evaluation

```bash
python evaluate.py \
  --config configs/mosi.yaml \
  --checkpoint runs/mosi/best.pt \
  --data-path /path/to/MOSI/Processed/unaligned_50.pkl \
  --drop-modality audio
```

Noise:

```bash
python evaluate.py \
  --config configs/mosi.yaml \
  --checkpoint runs/mosi/best.pt \
  --data-path /path/to/MOSI/Processed/unaligned_50.pkl \
  --noise-modality vision \
  --noise-std 1.0
```

---

## 8. Useful ablations

No external memory:

```bash
python train.py --config configs/mosi.yaml --data-path DATA.pkl \
  --output runs/mosi_no_memory \
  --set memory.enabled=false
```

Static initialized memory (no online consolidation):

```bash
python train.py --config configs/mosi.yaml --data-path DATA.pkl \
  --output runs/mosi_static_memory \
  --set memory.online_update=false
```

Ablate novelty from write score:

```bash
--set memory.write_novelty_power=0.0
```

Ablate stability:

```bash
--set memory.write_stability_power=0.0
```

Ablate hardness:

```bash
--set memory.write_hardness_power=0.0
```

Ablate utility/redundancy pressure approximately by:

```bash
--set memory.redundancy_weight=0.0
```

Fast writer instead of a slow EMA writer:

```bash
--set memory.writer_momentum=0.0
```

---

## 9. Configuration overrides

Any nested YAML field can be changed without editing the file:

```bash
python train.py --config configs/mosi.yaml --data-path DATA.pkl \
  --set model.d_model=192 \
  --set memory.num_regimes=10 \
  --set memory.slots_per_regime=48 \
  --set train.lr=0.0001
```

Values are parsed as YAML, so booleans/numbers/lists work.

---

## 10. First experiments I recommend

Do **not** immediately spend weeks tuning the full model. Run this falsification ladder first:

1. `memory.enabled=false`
2. full RAMP
3. static memory (`memory.online_update=false`)
4. no novelty
5. no stability
6. fast writer (`writer_momentum=0`)
7. missing/noisy modalities
8. inspect low/mid/high interaction-difficulty gains

If the full method does not consistently improve the high-interaction subset over the base model, the main hypothesis needs revision before adding modules.

See `EXPERIMENT_PLAN.md` for the complete paper-facing matrix.
