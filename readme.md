# Diagnosing and Repairing Shortcut Learning in Person Re-Identification Models for Robotic Perception

While building a person-tracking system at IDOLL Robotics, a companion robot startup, I noticed the model's matching accuracy collapsed when subjects changed clothes between camera views. Investigating why led to this project — a full mechanistic audit of what the model had actually learned. The answer: it was tracking shirts, not people.

The fix generalizes far beyond robotics. The same shortcut problem appears in any re-ID system deployed where appearance is inconsistent — and nowhere is that more critical than wildlife monitoring. Large conservation parks across Africa need autonomous systems that can track individual animals across days, seasons, and terrain. An animal's coat doesn't change, but lighting does — dramatically. A model that shortcuts on color temperature or shadow pattern will fail in the field the same way this one fails under a hue shift. Building re-ID systems that track identity, not appearance artifacts, is the foundation that makes that possible.

A six-phase mechanistic audit of shortcut learning in a standard ResNet50 re-ID baseline trained on Market-1501.
The project diagnoses color as a primary shortcut feature, quantifies its severity, and repairs it through targeted augmentation.

**Dataset:** Market-1501 | **Backbone:** ResNet50 | **Framework:** torchreid

---

## Phase 1 — Baseline Training

Standard ResNet50 trained on Market-1501 for 60 epochs with triplet + softmax loss.

| Metric | Result |
|--------|--------|
| mAP | 65.3% |
| Rank-1 | 82.8% |
| Rank-5 | 93.0% |
| Rank-10 | 95.0% |
| Rank-20 | 97.1% |

Consistent with published ResNet50 baselines on Market-1501 (~65.9% mAP in comparable setups).
Standard hyperparameters were chosen deliberately to represent a typical practitioner baseline.

---

## Phase 2 — Mechanistic Interpretation

Linear probes trained on layer activations to measure how much color and identity information each layer carries.

| Layer | Color Accuracy | Identity Accuracy |
|-------|---------------|-------------------|
| layer1 | 90.4% | 7.1% |
| layer2 | 91.5% | 10.1% |
| layer3 | 92.7% | 17.9% |
| layer4 | 92.9% | 100.0% |

**Finding:** A simple linear classifier achieves 92.9% clothing color accuracy on layer4 embeddings —
the same embeddings used for re-ID matching. Color is a primary linearly separable feature
throughout the network, consistent with shortcut learning. The 100% identity accuracy at layer4
reflects memorization of training identities.

---

## Phase 3 — Causal Ablation

Top color-predictive channels identified via linear probe coefficients, then zeroed at inference.

| Intervention | mAP | Rank-1 | Change |
|---|---|---|---|
| Baseline | 65.3% | 82.8% | — |
| Layer4 top 50 zeroed | 66.5% | 83.4% | +1.2%, +0.6% |
| Layer4 top 10 zeroed | 66.5% | 83.3% | +1.2%, +0.5% |
| Layer1 top 10 zeroed | 66.5% | 83.4% | +1.2%, +0.6% |

**Finding:** Zeroing color-predictive channels consistently *improves* accuracy regardless of
intervention depth or size. Color acts as distributed noise across all layers — removing it
forces the model to rely on more genuine identity features. The identical improvement magnitude
at layer1 and layer4 confirms the shortcut is systemic, not localized.

---

## Phase 4 — Stress Test

Query images recolored via HSV torso hue shift using `ColorShiftDataset`.
Gallery images unchanged — simulating a person wearing different clothes.

| Hue Shift | mAP | Rank-1 | mAP Drop |
|-----------|-----|--------|----------|
| Baseline | 65.3% | 82.8% | — |
| 30° | 32.8% | 40.9% | -32.5% |
| 90° | 23.1% | 27.9% | -42.2% |
| 180° | 49.2% | 62.7% | -16.1% |

**Finding:** Model accuracy collapses under clothing color change, with worst-case degradation
at 90° hue shift — the point of maximum color distance from the training distribution.
The partial recovery at 180° confirms the shortcut is tied to specific hue values seen during
training, not sensitivity to color change in general.

**Key insight:** Phase 3 and Phase 4 appear to contradict each other — zeroing color channels
*improves* accuracy, yet changing query colors *collapses* it. Both are true simultaneously.
Symmetric removal (Phase 3) helps because color was noise hurting same-person consistency.
Asymmetric shifting (Phase 4) breaks matching because the model uses color to link query to
gallery — when they no longer match, the wrong person is retrieved.

---

## Phase 5 — Repair

**Approach:** Custom `TorsoColorJitter` augmentation injected into the training pipeline via monkey patching.
Randomly shifts torso hue by ±`max_hue_shift` degrees per image during training.

### Repair v1 — Fine-tuning from Phase 1 checkpoint (max_hue_shift=30°)

| Metric | Baseline | Repaired | Change |
|--------|----------|----------|--------|
| Standard mAP | 65.3% | 63.8% | -1.5% |
| mAP under 90° shift | 23.1% | 22.0% | -1.1% |

**Finding:** Fine-tuning from a shortcut-heavy checkpoint fails. The color shortcut is too
deeply encoded across all four layers to be overwritten by 60 epochs of mild augmentation.

### Repair v2 — Training from scratch (max_hue_shift=90°)

| | Standard mAP | Standard Rank-1 | mAP (90° shift) | Rank-1 (90° shift) |
|--|--|--|--|--|
| Baseline | 65.3% | 82.8% | 23.1% | 27.9% |
| Repair v2 | 65.5% | 82.0% | **44.7%** | **58.7%** |

### Full Severity Curve — Baseline vs Repaired

| | Standard | 30° shift | 90° shift | 180° shift |
|--|--|--|--|--|
| Baseline | 65.3% | 32.8% | 23.1% | 49.2% |
| Repaired | 65.5% | 57.3% | 44.7% | 57.2% |
| Improvement | +0.2% | +24.5% | +21.6% | +8.0% |

**Finding:** Training from scratch with strong color augmentation matches baseline standard
accuracy while nearly doubling robustness under color shift. The repaired model's severity
curve is dramatically flatter — degrading from 65.5% to 44.7% versus the baseline's cliff
from 65.3% to 23.1%. In deployment, this is the difference between graceful degradation
and catastrophic failure when a tracked subject changes clothes.

---

## Summary

When color information is removed symmetrically, the model adapts and slightly improves.
When color consistency is broken asymmetrically, the model catastrophically fails.
Training from scratch with torso color jitter eliminates the cliff without sacrificing
standard accuracy — the repaired model learns to track identity, not shirts.

| Phase | Key Finding |
|-------|-------------|
| Phase 2 | Color is linearly readable at 92.9% accuracy in the final embedding |
| Phase 3 | Color channels are noise — removing them helps rather than hurts |
| Phase 4 | Color shift causes 42.2 mAP point drop at 90° — catastrophic failure |
| Phase 5 | Training from scratch with ±90° jitter recovers robustness at no accuracy cost |