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

