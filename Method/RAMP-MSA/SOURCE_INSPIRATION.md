# Source Inspiration Notes: Sim-MPNet -> RAMP-MSA

The uploaded Sim-MPNet source was used as a **conceptual reference**, not copied into this package.

The main source-level ideas carried forward are:

| Sim-MPNet source idea | RAMP adaptation |
|---|---|
| persistent `concept_pool` outside normal gradient parameters | persistent key-policy procedural memory |
| prototype/core used to route features to a local memory cluster | interaction-regime core used for coarse routing |
| `theta` reader + EMA-style `phi_k` writer | fast query encoder + slow EMA writer |
| memory residual plus current-context residual | memory policy correction plus current base fusion |
| online memory replacement | hardness/novelty/stability/regret-based consolidation |

Important source-level behaviors intentionally **not** copied:

- image-specific 2D convolutions and DS-GIM;
- category-count-based memory design;
- direct feature-vector storage as the memory value;
- previous-batch vs current-batch loss difference as the memory update controller;
- fixed image-stage shapes/resolution assumptions.

The core research transformation is therefore:

**Sim-MPNet:** remember representative category features.

**RAMP-MSA:** remember representative *loss-reducing fusion procedures* for recurring multimodal interaction regimes.
