# RAMP Research Story

## Working title

**Remember How to Fuse: Regret-Aware Multimodal Procedural Memory for Multimodal Sentiment Analysis**

## 1. Problem shift

Most MSA methods ask one of four questions:

1. how to build cross-modal interactions;
2. how to separate shared/private information;
3. how to estimate modality confidence/conflict;
4. how to retrieve or learn semantic prototypes.

The proposed question is different:

> **Can fusion behavior itself be reusable knowledge?**

A difficult MSA sample is not difficult only because its semantic content is rare. It can be difficult because the *relationship* among text, audio and vision is unusual: text may dominate, audio may reverse lexical polarity, visual cues may rescue ambiguous language, or all three modalities may conflict.

A parametric fusion network must compress all such cases into weights. A sample-retrieval system retrieves what happened before. RAMP instead externalizes **what fusion correction was useful before**.

This changes external memory from declarative memory (*what did I see?*) to procedural memory (*what should I do in this interaction regime?*).

---

## 2. What is inherited from Sim-MPNet — and what is not

The useful transferable principle from Sim-MPNet is not its image-specific DS-GIM block. The transferable principle is:

- a persistent external state outside ordinary gradient-updated parameters;
- similarity-based routing to relevant memory;
- a slow/EMA writer that stabilizes the memory coordinate system;
- memory information used as a residual together with current evidence;
- online memory refresh rather than a fixed prototype table.

RAMP changes the object being remembered. It does **not** make one sentiment-class centroid bank. It remembers low-dimensional **fusion policies** indexed by multimodal interaction states.

It also replaces Sim-MPNet's batch-to-batch loss heuristic with a paired counterfactual quantity on the *same sample*:

`memory gain = base task loss - memory-corrected task loss`.

This is the direct signal used to judge whether a memory entry was actually useful.

---

## 3. Module I — Möbius Interaction State

Let the modality set be `M={T,A,V}`. A single shared coalition-fusion function is evaluated on all non-empty subsets:

`h_T, h_A, h_V, h_TA, h_TV, h_AV, h_TAV`.

Möbius inversion gives:

- `I_T = h_T`
- `I_A = h_A`
- `I_V = h_V`
- `I_TA = h_TA - h_T - h_A`
- `I_TV = h_TV - h_T - h_V`
- `I_AV = h_AV - h_A - h_V`
- `I_TAV = h_TAV - h_TA - h_TV - h_AV + h_T + h_A + h_V`

The decomposition isolates main effects, pair interactions, and the irreducible three-way interaction.

A learned state encoder converts these seven components into an **interaction-state key**. This is not a modality-confidence vector and not simply pairwise disagreement. The memory address explicitly contains higher-order non-additive interactions.

Important conceptual consequence:

> weak audio/visual modalities do not need to be strong standalone classifiers. They can be valuable **memory-address signals**, identifying which fusion regime the current sample resembles.

---

## 4. Module II — Interaction-addressed Procedural Memory

Each memory slot is a triple:

`(key_j, policy_j, utility_j)`.

### Key: when

The key is an EMA-writer embedding of the current Möbius interaction state. It describes the regime in which a fusion skill is useful.

### Policy: how

The policy is seven-dimensional, aligned with:

`[T, A, V, TA, TV, AV, TAV]`.

It tells the model which interaction components should be increased or decreased.

### Utility: whether it has actually helped

Utility is an EMA of paired task-loss reductions credited through the memory retrieval attention.

The memory is organized into unsupervised **interaction regimes**, not sentiment classes. Warm-up keys are clustered with spherical k-means; each regime stores several heterogeneous fusion policies.

Retrieval is hierarchical:

1. soft top-R routing to interaction-regime cores;
2. key similarity within the selected regimes;
3. weighted policy retrieval.

The retrieved policy is a residual correction. The current fused representation always remains present.

---

## 5. Where the policy supervision comes from

The strongest part of the method is that the desired memory value is not manually defined.

Use the seven normalized Möbius components as seven action directions. Introduce temporary coefficients `alpha in R^7`:

`h(alpha) = h_base + sum_r alpha_r * direction_r`.

At `alpha=0`, calculate the gradient of the current task loss with respect to alpha:

`p* = normalize(- dL / d alpha)`.

This is a **sample-specific local loss-reducing fusion policy**:

- positive coefficient: locally amplify that interaction coordinate;
- negative coefficient: suppress it.

The gradient is taken only with respect to temporary alpha. It is detached and never requires a second-order training graph.

For MOSI/MOSEI the task loss is regression; for IEMOCAP it is classification. The definition of the policy is therefore **label-space agnostic**.

This is an important difference from storing sentiment prototypes.

---

## 6. Module III — Regret-guided Memory Consolidation

The memory has limited capacity. The central learning question is therefore not merely how to write, but **what deserves to survive**.

### Candidate write factors

- **Hardness**: base task loss. Easy, already-solved samples have low priority.
- **Novelty**: distance from existing keys in the routed interaction regime.
- **Stability**: cosine agreement of two oracle policies computed under feature-dropout views. High-loss but unstable/noisy examples should not dominate memory.

The default write score multiplies these three factors.

### Paired memory gain

For the same input and same deterministic head:

- `L_base`: no memory correction;
- `L_mem`: retrieved memory correction.

Define:

`G = L_base - L_mem`.

Unlike comparing losses of two different batches or epochs, `G` directly measures whether memory helped this sample.

Attention-weighted `G` updates each slot's historical utility.

### Assimilation vs accommodation

If a candidate is very similar to an existing slot, RAMP merges it with the slot (assimilation).

If it is novel, RAMP inserts it by evicting a low-retention slot. Retention combines historical utility and redundancy, so low-utility duplicate skills are forgotten first.

### Regime-specific plasticity

Each regime tracks EMA hardness, novelty and memory gain. Hard/novel/poorly-served regimes receive a larger update budget; already-useful regimes are changed more conservatively.

---

## 7. Why this is not the closest recent work

The current MSA frontier is crowded, so the paper must explicitly separate its question from nearby methods.

### PaSE — AAAI 2026

**Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis** uses class prototypes for representation calibration/alignment and Shapley-based optimization balancing.

RAMP's memory is not a class centroid and is not used for gradient balancing. It stores sample-derived *fusion actions* indexed by high-order interaction states.

Official page: https://ojs.aaai.org/index.php/AAAI/article/view/40355

### CICA — CVPR 2026

**CICA** estimates modality confidence and uses confidence-informed attention for robust fusion.

RAMP is not a confidence-weighting method. Its interaction state includes non-additive pair/triple effects and is an address into historical procedural memory.

Official page: https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_CICA_Coupling_Confidence-Aware_Pretraining_with_Confidence-Informed_Attention_for_Robust_Multimodal_CVPR_2026_paper.html

### EBMC — CVPR 2026

**Enhance-then-Balance Modality Collaboration** targets modality competition through representation enhancement, energy-guided coordination and instance trust.

RAMP's core contribution is the externalization and retrieval of reusable fusion behavior, not an optimization-balancing objective.

Official page: https://openaccess.thecvf.com/content/CVPR2026/html/He_Enhance-then-Balance_Modality_Collaboration_for_Robust_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html

### CACR — CVPR 2026

**Conflict-Aware Adaptive Cross-Reconstruction** explicitly models emotional conflict and changes cross-reconstruction weights.

RAMP does not treat conflict score as the final fusion controller. Conflict/high-order interaction is used to retrieve a historical fusion skill.

Official page: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Conflict-Aware_Adaptive_Cross-Reconstruction_for_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html

### MMRest — CVPR 2026

**Multi-Metric Representation Learning Strategy Based on Clustering for Fine-Grained MSA** uses clustering and global/local metric learning to improve sentiment-space geometry.

RAMP uses clustering only to organize memory addresses. Its output is a retrieved action/policy rather than a learned sentiment metric.

Official paper: https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_Multi-Metric_Representation_Learning_Strategy_Based_on_Clustering_for_Fine-Grained_Multimodal_CVPR_2026_paper.pdf

### Prototype-as-Prompt — CVPR 2026

This method learns sentiment-semantic multimodal prototypes as soft prompts for LLMs.

RAMP memory values are not latent semantic prompts. They are task-loss-derived procedural vectors that say how to change interaction contributions.

Official page: https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Prototype-as-Prompt_Multimodal_Sentiment_Prototypes_Endowing_Large_Language_Models_the_Capability_CVPR_2026_paper.html

### Retrieval-augmented / cross-sample MSA

Recent work has already shown that retrieving semantically related samples can help MSA, including:

- *Towards Multimodal Sentiment Analysis via Contrastive Cross-modal Retrieval Augmentation and Hierarchical Prompts* (arXiv 2025 / T-AFFC 2026): https://arxiv.org/abs/2508.07666
- *Hyper-Modality Enhancement for Multimodal Sentiment Analysis with Missing Modalities* (NeurIPS 2025): https://proceedings.neurips.cc/paper_files/paper/2025/hash/d079de28c5e3f22e3507db24e870126b-Abstract-Conference.html

This is exactly why RAMP must **not** be framed as sample retrieval. It retrieves compressed fusion policies rather than historical semantic evidence.

---

## 8. Main hypothesis and falsification test

The paper should not rely only on aggregate SOTA numbers.

Define interaction difficulty from the norms of pairwise/triple Möbius components. Partition the test set into low/mid/high interaction groups.

Core hypothesis:

> The improvement of procedural memory over its own base branch should increase with interaction difficulty.

For regression, report:

`DeltaMAE = BaseMAE - RAMP_MAE`

for each bucket.

For classification, report the analogous accuracy/F1 gain.

If the high-interaction subset does not benefit more, the main explanation is not supported even if mean benchmark performance improves slightly.

---

## 9. Claims that are safe only after experiments

Do not write these as facts before the results exist:

- RAMP is SOTA.
- procedural memory solves text dominance.
- interaction regimes correspond exactly to sarcasm or named linguistic phenomena.
- regret consolidation is always superior to FIFO/EMA memory.
- the method generalizes beyond MSA.

The code is structured to test these claims rather than assume them.

---

## 10. Target contribution statement if experiments succeed

1. **A new problem formulation:** multimodal fusion behavior is treated as reusable procedural knowledge rather than being stored only implicitly in parameters or retrieved as semantic examples.
2. **A coherent representation/action construction:** Möbius interaction decomposition simultaneously provides a high-order memory address and an interpretable seven-dimensional fusion action space.
3. **A task-derived memory target:** sample-specific fusion skills are generated from local task-loss gradients without test-time labels or gradients.
4. **A regret-aware continual memory mechanism:** external memory is consolidated according to hardness, novelty, stability, historical loss reduction and redundancy.
5. **Evidence beyond average benchmark scores:** gains are analyzed under high-conflict/high-interaction, noisy/missing-modality, low-resource and long-tail regimes.
