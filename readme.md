## Phase 1 Baseline Results
- mAP: 58.8%
- Rank-1: 79.0%
- Rank-5: 90.9%
- Rank-10: 94.2%
- Model: ResNet50, Market-1501, 60 epochs

## Phase 1 Baseline Results (Run 2)
- mAP: 64.3%
- Rank-1: 82.8%
- Rank-5: 92.9%
- Rank-10: 95.3%
- Rank-20: 96.9%
- Model: ResNet50, Market-1501, 60 epochs, softmax+triplet loss

## Phase 1 Baseline Results(Run 3)
- mAP: 65.3%
- Rank-1: 82.8%
- Rank-5: 93.0%
- Rank-10: 95.0%
- Rank-20: 97.1%
- Model: ResNet50, Market-1501, 60 epochs, triplet loss + softmax


## Phase 2 — Linear Probe Results

| Layer  | Color Accuracy | Identity Accuracy |
|--------|---------------|-------------------|
| layer1 | 90.4%         | 7.1%              |
| layer2 | 91.5%         | 10.1%             |
| layer3 | 92.7%         | 17.9%             |
| layer4 | 92.9%         | 100.0%            |

**Finding:** A linear classifier achieves 92.9% clothing color accuracy on 
layer4 embeddings — the same embeddings used for re-ID matching. Color is a 
primary linearly separable feature in the model's final representation across 
all four layers. This is consistent with shortcut learning.


## Phase 3 — Causal Ablation Results

### Layer4 Top 50 Channels Ablated
| Metric | Baseline | Ablated | Change |
|--------|----------|---------|--------|
| mAP    | 65.3%    | 66.5%   | +1.2%  |
| Rank-1 | 82.8%    | 83.4%   | +0.6%  |

**Finding:** Zeroing the 50 most color-predictive channels in layer4 
*improved* re-ID accuracy slightly. This suggests those channels were 
carrying color as noise that actively hurt matching performance — 
removing them forced the model to rely on more genuine identity features.
Next: ablate top 10 channels to test whether a smaller intervention 
produces a cleaner causal signal

### Ablation Summary
| Intervention          | mAP   | Rank-1 |
|-----------------------|-------|--------|
| Baseline              | 65.3% | 82.8%  |
| Layer4 top 50 zeroed  | 66.5% | 83.4%  |
| Layer4 top 10 zeroed  | 66.5% | 83.3%  |
| Layer1 top 10 zeroed  | 66.5% | 83.4%  |

**Finding:** Color acts as distributed noise across all network layers. 
Removing color-predictive channels improves accuracy consistently 
regardless of intervention depth or size.The model adapted to using the remaining 1998 channels for matching — which happened to be slightly better at encoding actual identity features. Hence the small improvement.

## Phase 4 — Stress Test Results

| Hue Shift | mAP   | Rank-1 | mAP Drop | Rank-1 Drop |
|-----------|-------|--------|----------|-------------|
| Baseline  | 65.3% | 82.8%  | —        | —           |
| 30°       | 32.8% | 40.9%  | -32.5%   | -41.9%      |
| 90°       | 23.1% | 27.9%  | -42.2%   | -54.9%      |
| 180°      | 49.2% | 62.7%  | -16.1%   | -20.1%      |

**Finding:** Model accuracy collapses under clothing color change,
with worst-case degradation at 90° hue shift (maximum color distance
from training distribution). The 180° recovery confirms the shortcut
is tied to specific hue values seen during training, not just
sensitivity to color change in general.


## Summary of the Phase 1 through Phase 4
When color information is removed symmetrically, the model adapts and slightly improves. When color consistency is broken asymmetrically, the model catastrophically fails — revealing that it over-relies on color as a matching signal rather than learning robust identity features