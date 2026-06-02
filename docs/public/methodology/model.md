# Model — VGG16 with Transfer Learning

> Reference document — describes the model architecture and the transfer-learning
> strategy used in [Phase 1](../phases/phase-1-vgg16-baseline.md).
> Phase 2 will extend this to ResNet50; the strategy described here generalizes.

---

## Why VGG16?

[Wong et al. (2025, PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322624) report **99.24% accuracy** on this exact dataset using VGG16 with transfer learning. Our objective in Phase 1 is to **replicate their methodology** to validate our pipeline before extending to multi-architecture comparison in Phase 2.

VGG16 is also a well-suited backbone for the downstream XAI work in Phase 3:

- Its sequential block structure makes **Grad-CAM** straightforward (the last convolutional block has a clean 7×7 spatial resolution)
- Its widespread pretrained availability ensures **reproducibility**
- Its simplicity (compared to attention-based models) avoids confounding the XAI comparison with attention artifacts

---

## VGG16 architecture overview

VGG16 (Simonyan & Zisserman, 2014) is a CNN with **13 convolutional layers** organized in 5 blocks, followed by 3 fully-connected layers:

```
Input (3 × 224 × 224)
│
├── Block 1: Conv 64 × 2 → MaxPool         (output: 64 × 112 × 112)
├── Block 2: Conv 128 × 2 → MaxPool        (output: 128 × 56 × 56)
├── Block 3: Conv 256 × 3 → MaxPool        (output: 256 × 28 × 28)
├── Block 4: Conv 512 × 3 → MaxPool        (output: 512 × 14 × 14)
├── Block 5: Conv 512 × 3 → MaxPool        (output: 512 × 7 × 7)   ← we fine-tune this
│
└── Classifier (FC):
    Flatten (25088)
    Linear (25088 → 4096) → ReLU → Dropout
    Linear (4096 → 4096)  → ReLU → Dropout
    Linear (4096 → 1000)                   ← we replace this
```

Total parameters: **~138 million**. Pretrained weights on ImageNet (1000-class natural image dataset, ~528 MB).

### How convolution and pooling actually work

A **convolution filter** is a small grid of weights (e.g. 3×3) that slides across the image. At each position it multiplies-and-sums the pixels beneath it; a high output means *"the pattern this filter detects is present here."* Sliding it across the whole image produces a **feature map** — a new image marking where that pattern occurs. Because the same filter is applied everywhere, a feature is detected regardless of *where* it appears (translation invariance) — useful when a tumor can sit anywhere in the brain. The network is not given these filters; it **learns** them during training (here, the early-block filters arrive pretrained from ImageNet).

"Conv 64" means **64 different filters** run in parallel, producing **64 feature maps (channels)** — one per pattern (vertical edges, textures, etc.). Each convolution is followed by **ReLU**, which zeroes negative values (keep *how present*, discard *how absent*), giving the network its non-linearity.

**MaxPool** then halves the spatial size by keeping the maximum value in each 2×2 neighborhood — *"was this pattern present anywhere nearby? keep the strongest evidence."* This cuts computation and makes the network robust to small shifts of the input.

*Input note:* VGG16 expects 3 (RGB) channels, but MRI is grayscale (1 channel). The single grayscale channel is **replicated three times** to fit the pretrained input — acceptable because the network keys on shape and texture, not color.

### Why spatial size shrinks while depth grows

In the block table above, two opposite trends run in parallel: spatial resolution shrinks (224 → 7) while channel depth grows (3 → 512). They have different causes but share one logic.

**Space shrinks** because the more abstract a feature is, the fewer positions you need to localize it — like zooming a street map out to a country map: far fewer points, yet each point means much more. Pooling also enlarges each deeper neuron's *receptive field*, so a block-5 cell "sees" a large region of the brain — big enough to recognize a whole tumor.

**Depth grows** because the number of useful patterns explodes with complexity: there are only a handful of basic edges, but a vast number of high-level structures ("rounded bright peripheral mass," "ring with a dark center," "symmetric normal ventricles"). More concepts to detect → more filters.

Together, the network progressively **trades "where exactly" for "what exactly."** The final `512 × 7 × 7` tensor reads as: *"for each of 49 regions, which of 512 high-level concepts are present?"* — enough for the classifier head to decide the class, and the reason the Grad-CAM heatmaps in Phase 3 are coarse 7×7 maps (see [`clinical-context.md`](clinical-context.md) for the per-class hypotheses this enables).

---

## Transfer learning strategy

We do **not** train VGG16 from scratch. The dataset is too small (~5,700 training images) and the cost would be prohibitive. Instead, we **transfer** the features VGG16 already learned on ImageNet.

### Two-stage protocol (replicates Wong et al.)

The training proceeds in two stages, both lasting 50 epochs.

#### Stage 1 — Head-only training

```
Backbone (Conv blocks 1–5)   FROZEN  (no gradient updates)
Custom classifier head       TRAINING (Adam, lr = 1e-3)
```

The convolutional backbone is frozen — its weights stay fixed at the ImageNet checkpoint. Only the **custom classifier head** trains. This head replaces the original 3 FC layers with:

```
Flatten (25088)
Linear (25088 → 256) → ReLU → Dropout(0.5)
Linear (256 → 4)
```

The 4 outputs correspond to the 4 brain-tumor classes. No softmax is applied here — `CrossEntropyLoss` applies it internally.

**Why freeze the backbone in Stage 1?**
The pretrained backbone already extracts useful visual primitives (edges, textures, shapes). Training the head first lets it adapt to *what* the backbone already sees, without destroying those primitives. If we trained everything end-to-end from the start, the large gradient signal flowing back from the randomly-initialized head would corrupt the backbone before it had a chance to be useful.

#### Stage 2 — Fine-tune conv5 + head

```
Backbone Block 1–4           FROZEN
Backbone Block 5 (conv5)     TRAINING (lr = 1e-4)
Custom classifier head       TRAINING (lr = 1e-4)
```

After Stage 1 plateaus, we unfreeze the last convolutional block (conv5, layers 24–30 in the torchvision indexing). The head continues to train. Learning rate **drops 10×** to avoid destroying the now-converged Stage-1 features.

**Why unfreeze conv5 specifically?**
Conv5 has the largest receptive field — each neuron sees a large patch of the original image. Tumors typically occupy large image regions, so conv5 is where domain-specific features (tumor texture, tumor boundary patterns) live. Earlier blocks encode generic primitives (lines, gradients) that transfer perfectly from ImageNet and don't need updating.

This is a well-established pattern from the transfer-learning literature (Yosinski et al. 2014; Howard & Ruder 2018).

---

## Implementation

The model lives in [`src/neurolens/models/vgg16.py`](../../../src/neurolens/models/vgg16.py). The factory pattern in [`src/neurolens/models/factory.py`](../../../src/neurolens/models/factory.py) provides architecture dispatch — Phase 2 will add ResNet50 with a single line in the registry, no `if/elif` branching.

```python
from neurolens.models.factory import build_model

# Stage 1 — backbone frozen
model = build_model("vgg16", num_classes=4, stage=1)

# Transition to Stage 2 — same model instance, unfreeze conv5
from neurolens.models.vgg16 import unfreeze_conv5
unfreeze_conv5(model)
```

---

## Methodological deviations from Wong et al.

We document deviations explicitly for transparency.

| Aspect | Wong et al. (Keras) | NeuroLens (PyTorch) | Justification |
|--------|---------------------|---------------------|---------------|
| Input normalization | `rescale=1/255` (scale to [0, 1]) | ImageNet mean/std normalization | Standard for ImageNet-pretrained models in PyTorch; expected by torchvision weights |
| Head architecture | `Flatten → Dense(256, ReLU) → Dense(4, softmax)` | `Flatten → Linear(256) → ReLU → Dropout(0.5) → Linear(4)` | Same width, dropout added (Wong did not specify; 0.5 is convention) |
| Optimizer | Adam | Adam | Identical |
| Learning rates | 1e-3 (stage 1) / 1e-4 (stage 2) | Identical | Identical |
| Epochs | 50 per stage | 50 per stage | Identical |
| Augmentation | Shear 0.2, Zoom 0.2, Horizontal flip | Equivalent in `torchvision.transforms.v2` | Same semantics; different implementation |
| Train/val split | 80/10/10 single split | 5-fold stratified CV | More rigorous statistical estimate |

All deviations are minor and well-justified. The pipeline reproduces Wong's *methodology* rather than copying Wong's *code* (which was Keras; ours is PyTorch).

---

## What this model can and cannot do

**Can do:**
- Classify a 2D MRI slice into one of 4 tumor types with ~94–97% test accuracy (single fold; 5-fold mean reported in [Phase 1](../phases/phase-1-vgg16-baseline.md))
- Produce class probabilities (via softmax on logits) suitable for Grad-CAM, LIME, and SHAP (Phase 3)
- Be the building block for the multi-architecture comparison in Phase 2

**Cannot do:**
- Segment the tumor (where it is); only classify (what it is)
- Process 3D volumes; works on individual 2D slices
- Handle MRI scans with classes outside the training distribution (the model will confidently mispredict OOD inputs — a known XAI motivation)
- Replace a radiologist; intended as a research baseline, not a clinical tool

---

## References

- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv:1409.1556.
- Wong, Y. C., Choi, L. K., Singh Sandhu, R., Mohd Faizal Bin Abdullah, A., Yusuf, R., Saiful Bahari Bin Shaari, A., & Mohamad Shariff, S. F. F. (2025). *Classifying brain tumors on magnetic resonance imaging by deep learning techniques*. PLOS ONE 20(5): e0322624.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are features in deep neural networks?* NIPS.
- Howard, J., & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text Classification*. ACL. (cross-domain transfer-learning principles)
