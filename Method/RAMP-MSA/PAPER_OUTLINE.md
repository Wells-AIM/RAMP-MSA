# Paper Outline — Working Draft

## Title

**Remember How to Fuse: Regret-Aware Multimodal Procedural Memory for Multimodal Sentiment Analysis**

## Abstract skeleton

Multimodal sentiment analysis models usually encode fusion knowledge implicitly in network parameters, while recent retrieval/prototype methods externalize semantic evidence or class-level representations. We argue that another reusable form of knowledge is overlooked: **fusion procedure** — how modality interactions should be adjusted under recurring patterns of agreement, conflict and complementarity. We introduce RAMP, a Regret-Aware Multimodal Procedural Memory framework. RAMP evaluates a shared fusion function on all text/audio/visual coalitions and applies Möbius inversion to obtain unimodal, pairwise and three-way interaction components. These components define both an interaction-aware memory address and a seven-dimensional fusion action space. During training, RAMP derives a sample-specific local fusion policy from the task-loss gradient with respect to temporary interaction coefficients and stores these policies in a persistent key-policy memory. At inference, the current interaction state retrieves relevant policies and applies them as residual corrections without labels or test-time gradients. To keep a finite memory useful, we develop regret-aware consolidation based on sample hardness, policy novelty/stability, historical paired loss reduction and memory redundancy. Experiments should evaluate standard MOSI/MOSEI/IEMOCAP performance together with conflict, missing/noisy-modality, low-resource and interaction-difficulty analyses.

## Introduction logic

1. MSA has progressed from static fusion to cross-modal attention, representation decomposition, reliability/conflict modeling and prototype/retrieval methods.
2. These approaches still mainly store **what features mean**; even retrieval methods retrieve semantic/contextual evidence.
3. Difficult multimodal cases often recur at the level of **interaction structure**, not lexical content: text-nonverbal conflict, audio rescue, visual suppression, three-way ambiguity.
4. What should transfer across such samples is a local **fusion correction**.
5. Challenge 1: define a high-order, permutation-consistent interaction state rather than only confidence scores.
6. Challenge 2: obtain supervision for “how to fuse” without hand-designed rules.
7. Challenge 3: maintain a finite external memory without letting noise/redundancy dominate.
8. RAMP solves these with Möbius interaction coordinates, gradient-derived procedural policies and regret-guided memory consolidation.

## Contributions

- **Procedural-memory formulation for MSA.** External memory stores fusion actions rather than semantic examples/class centroids.
- **Möbius interaction address/action duality.** The same seven high-order interaction coordinates describe the current modality regime and form the interpretable fusion action space.
- **Task-derived local fusion supervision.** A sample-specific policy is generated from the local task-loss gradient with respect to temporary interaction coefficients; the mechanism works for both regression and classification.
- **Regret-aware continual consolidation.** Memory write/merge/eviction is driven by hardness, novelty, stability, paired memory gain and redundancy.
- **Mechanism-oriented evaluation.** Beyond aggregate benchmarks, the paper tests whether gain increases with interaction difficulty and under conflict/noise/missing-modality regimes.

## Key reviewer questions to answer experimentally

1. Why memory instead of a parametric MLP/hypernetwork that predicts the policy?
2. Why procedural policy instead of retrieving raw training examples?
3. Why Möbius interactions instead of confidence/disagreement vectors?
4. Does the memory help particularly on high-interaction/rare regimes?
5. Is online consolidation better than static, FIFO, random or EMA memory?
6. Does the method merely add capacity? Use parameter-matched baselines.
7. Is there any train/test memory leakage? Memory must be frozen at validation/test.
