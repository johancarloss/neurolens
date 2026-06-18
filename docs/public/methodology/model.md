# Model — Transfer Learning with VGG16 and ResNet50

> Reference document — describes the model architectures and the transfer-learning
> strategy shared by [Phase 1 (VGG16)](../phases/phase-1-vgg16-baseline.md) and
> [Phase 2 (ResNet50)](../phases/phase-2-architectures.md). The two architectures
> are trained with the **same** protocol so the comparison is fair.

---

## The shared anatomy: backbone and head

Both architectures used here are split into two parts with very different jobs. This split is the foundation of the whole transfer-learning strategy, so it is worth stating first.

```
        IMAGE (pixels)
            │
            ▼
   ┌──────────────────┐
   │     BACKBONE     │   the convolutional layers — they LOOK at the image and
   │   ("the body")   │   turn pixels into features ("edge here, bright blob there")
   └──────────────────┘
            │  a compact feature summary
            ▼
   ┌──────────────────┐
   │       HEAD       │   the final layers — they take the summary and DECIDE
   │ ("the decision") │   ("given these features → glioma")
   └──────────────────┘
            │
            ▼
     4 class scores (glioma · meningioma · notumor · pituitary)
```

- The **backbone** extracts features. This skill is *general* — an edge is an edge whether the image is a cat or a tumor — so we **reuse** the backbone that arrives pretrained on ImageNet rather than learning it from scratch.
- The **head** makes the task-specific decision. ImageNet's original head decided among 1000 everyday objects, which is useless to us, so we **discard it and bolt on a fresh head** with 4 outputs (our tumor classes).

This is why transfer learning works: keep the expensive, general backbone; replace and train only the cheap, specific head. The two-stage protocol below is built directly on this distinction.

---

## Why these two architectures?

**VGG16 (Phase 1).** [Wong et al. (2025, PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322624) report **99.24% accuracy** on this exact dataset using VGG16 with transfer learning. Replicating their methodology validates our pipeline before extending to a multi-architecture comparison.

**ResNet50 (Phase 2).** Chosen as a deliberately **opposite design** to VGG16, to test whether results depend on the architecture or on the data:

- **Philosophically opposite** — VGG16 just stacks convolutions; ResNet50 introduced *residual connections* (2016), the breakthrough that made very deep networks trainable (see below).
- **~5× fewer parameters** (~25M vs ~138M) yet often matches or beats VGG16 — a fairer test of "does depth/design matter here?".
- **The de-facto default** in medical imaging, which lends the comparison external legitimacy.
- **Grad-CAM compatible** — its last residual block (`layer4[-1]`) mirrors VGG16's last conv block, so the Phase 3 XAI comparison stays clean. (Attention-based models like ViT were excluded precisely because attention would confound that comparison.)

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

## ResNet50 architecture overview

ResNet50 (He et al., 2016) is a 50-layer CNN whose backbone is organized into four groups of **bottleneck blocks** (in torchvision: `layer1`–`layer4`). It produces a `2048 × 7 × 7` feature tensor, which a **global average pooling** step then reduces to a 2048-vector for the head.

```
Input (3 × 224 × 224)
│
├── stem: Conv 7×7 → MaxPool
├── layer1: 3 bottleneck blocks     (output: 256 × 56 × 56)
├── layer2: 4 bottleneck blocks     (output: 512 × 28 × 28)
├── layer3: 6 bottleneck blocks     (output: 1024 × 14 × 14)
├── layer4: 3 bottleneck blocks     (output: 2048 × 7 × 7)   ← we fine-tune this
│
├── Global Average Pooling          (output: 2048-vector)    ← key difference from VGG16
└── fc: Linear (2048 → 1000)                                 ← we replace this
```

Total parameters: **~25 million** — roughly 5× fewer than VGG16, despite being deeper. The reason for that paradox is the next two subsections.

> **Naming caution:** ResNet's backbone groups (`layer1`–`layer4`) are sometimes called "stages" in the literature. That is **not** the same as our *training* stages (Stage 1 / Stage 2 below). Here, "stage" always means a training phase; the backbone groups are called `layer1`–`layer4`.

### The residual idea (skip connections)

The motivation is a surprising failure of plain deep networks like VGG: stacking *more* layers eventually made them **worse**, even on the training set. This was not overfitting — it was a **degradation/optimization problem**. Counterintuitively, a deep stack struggled even to *preserve* a good signal, because making a sequence of layers learn the identity function (copy input → output unchanged) is hard.

ResNet's fix is small and elegant. Instead of asking a block of layers to produce the full transformation `H(x)`, ask it to produce only the **adjustment** `F(x)`, then add the original input back through a **skip connection**:

```
   plain (VGG-style):           residual (ResNet):

   x → [layers] → H(x)          x ──────────────┐ (skip / identity)
                                │                │
                                ▼                │
                             [layers] → F(x)     │
                                │                │
                                ▼                │
                               (+) ◄─────────────┘
                                │
                                ▼
                          output = F(x) + x
```

Why this helps:

- **"Do nothing" becomes trivial.** If extra layers don't help, the block just learns `F(x) = 0`, and the output is `0 + x = x` — the input passes through untouched. Learning to output zeros is easy (weights are pulled toward zero by initialization and weight decay); learning to copy perfectly is hard. ResNet inverted the difficulty.
- **When a block *should* help**, it only learns the *delta* over what already arrived — a smaller, more stable target than recomputing everything (like editing a document with a patch versus rewriting it).
- **The skip is also a gradient highway.** During backpropagation the gradient flows back through the `+x` shortcut without diluting across all the layers, which is why ResNet can be 50+ layers deep where VGG plateaus around 19.

### Global average pooling and why the head is small

After the backbone, VGG16 and ResNet50 hand the features to the head in opposite ways:

- **VGG16 flattens** the `512 × 7 × 7` tensor into **25,088** numbers — it keeps *every spatial position*.
- **ResNet50 averages** each of its 2048 feature maps down to a **single number** (global average pooling), giving **2,048** numbers — it keeps *how much* of each feature exists anywhere in the image, and discards *where*.

For classification ("is there a tumor, and which?") "how much" is sufficient — you don't need the exact coordinate to assign a label. This is the same "trade *where* for *what*" idea from the VGG section, pushed to its conclusion. It has two consequences:

1. **Parameter count.** The head's first layer needs one weight per incoming number. VGG16 feeds it 25,088 (so it needs a wide 256-unit bottleneck to tame them: ~6.4M weights); ResNet50 feeds it only 2,048 (so it goes straight to the 4 classes: ~8K weights). This single difference is why ResNet50 has ~5× fewer parameters and trains faster (~50 vs ~62 min/fold in our runs).
2. **The heads differ by design — not by accident.** ResNet50's pooling already condensed the features, so forcing a VGG-style 256-unit bottleneck onto it would redo work the pooling already did, and could hurt. Each head is shaped to fit what its backbone delivers. (This is decision D3 in the Phase 2 plan, and is locked by a unit test so nobody "fixes" it.)

> *XAI note:* because ResNet50's pooling discards spatial location, the Phase 3 Grad-CAM must *recover* "where" from the last conv block (`layer4[-1]`) — the same role `features[-1]` plays for VGG16.

### VGG16 vs ResNet50 at a glance

| Aspect | VGG16 | ResNet50 |
|--------|-------|----------|
| Core idea | stack convolutions | residual (skip) connections |
| Depth | 16 layers (~19 with FC) | 50 layers |
| Backbone → head bridge | flatten (25,088) | global average pooling (2,048) |
| Custom head | `Flatten → Linear(256) → ReLU → Dropout → Linear(4)` | `Dropout → Linear(2048 → 4)` |
| Stage-2 unfrozen block | `conv5` (Block 5) | `layer4` |
| Grad-CAM target (Phase 3) | `features[-1]` | `layer4[-1]` |
| Parameters | ~138M | ~25M |

---

## Transfer learning strategy

We do **not** train from scratch. The dataset is small (~5,600 training images) and the cost would be prohibitive. Instead we **transfer** the features the backbone already learned on ImageNet, in two stages. This protocol is **identical for both architectures** — only the name of the block unfrozen in Stage 2 differs.

### Two-stage protocol (replicates Wong et al.)

Both stages last 50 epochs.

#### Stage 1 — Head-only training

```
Backbone                     FROZEN  (no gradient updates)
Custom classifier head       TRAINING (Adam, lr = 1e-3)
```

The backbone is frozen — its weights stay at the ImageNet checkpoint. Only the **custom head** trains. For VGG16 the head is:

```
Flatten (25088) → Linear(256) → ReLU → Dropout(0.5) → Linear(4)
```

and for ResNet50 it is the smaller `Dropout(0.5) → Linear(2048 → 4)` (because pooling already flattened the features — see above). The 4 outputs correspond to the 4 tumor classes; no softmax is applied (`CrossEntropyLoss` applies it internally).

**Why freeze the backbone in Stage 1?**
The pretrained backbone already extracts useful visual primitives (edges, textures, shapes). Training the head first lets it adapt to *what* the backbone already sees, without destroying those primitives. If we trained everything end-to-end from the start, the large gradient signal flowing back from the randomly-initialized head would corrupt the backbone before it had a chance to be useful.

#### Stage 2 — Fine-tune the last backbone block + head

```
Backbone (all but the last block)   FROZEN
Backbone last block                 TRAINING (lr = 1e-4)   ← conv5 (VGG16) / layer4 (ResNet50)
Custom classifier head              TRAINING (lr = 1e-4)
```

After Stage 1 plateaus, we unfreeze the **last** convolutional block — `conv5` for VGG16, `layer4` for ResNet50 — while the head keeps training. The learning rate **drops 10×** to avoid destroying the now-converged Stage-1 features.

**Why the last block specifically?**
It has the largest receptive field — each neuron sees a large patch of the original image. Tumors occupy large regions, so this is where domain-specific features (tumor texture, boundary patterns) live. Earlier blocks encode generic primitives (lines, gradients) that transfer perfectly from ImageNet and don't need updating. This is a well-established pattern (Yosinski et al. 2014; Howard & Ruder 2018).

The factory exposes this transition uniformly: `unfreeze_for_stage2(model, arch)` dispatches to the right block for each architecture, so the training loop never needs to know which model it is running.

---

## Implementation

The architectures live in [`src/neurolens/models/vgg16.py`](../../../src/neurolens/models/vgg16.py) and [`src/neurolens/models/resnet50.py`](../../../src/neurolens/models/resnet50.py). The factory in [`src/neurolens/models/factory.py`](../../../src/neurolens/models/factory.py) provides architecture dispatch — adding an architecture is one entry per registry, with no `if/elif` branching in the training code.

```python
from neurolens.models.factory import build_model, unfreeze_for_stage2

# Stage 1 — backbone frozen, only the head trains
model = build_model("resnet50", num_classes=4, stage=1)   # or "vgg16"

# Transition to Stage 2 — same model instance, unfreeze the last block
unfreeze_for_stage2(model, "resnet50")                     # conv5 for vgg16, layer4 for resnet50
```

---

## Methodological deviations from Wong et al.

These apply to the **VGG16** replication (Wong did not use ResNet50). We document them for transparency.

| Aspect | Wong et al. (Keras) | NeuroLens (PyTorch) | Justification |
|--------|---------------------|---------------------|---------------|
| Input normalization | `rescale=1/255` (scale to [0, 1]) | ImageNet mean/std normalization | Standard for ImageNet-pretrained models in PyTorch; expected by torchvision weights |
| Head architecture | `Flatten → Dense(256, ReLU) → Dense(4, softmax)` | `Flatten → Linear(256) → ReLU → Dropout(0.5) → Linear(4)` | Same width, dropout added (Wong did not specify; 0.5 is convention) |
| Optimizer | Adam | Adam | Identical |
| Learning rates | 1e-3 (stage 1) / 1e-4 (stage 2) | Identical | Identical |
| Epochs | 50 per stage | 50 per stage | Identical |
| Augmentation | Shear 0.2, Zoom 0.2, Horizontal flip | Equivalent in `torchvision.transforms.v2` | Same semantics; different implementation |
| Train/val split | 80/10/10 single split | 5-fold stratified CV | More rigorous statistical estimate |

ResNet50 reuses the **same hyperparameters** as VGG16 (Stage 1 lr 1e-3, Stage 2 lr 1e-4, 50 epochs, same augmentation, same seed) so the Phase 2 comparison isolates the architecture as the only changed variable.

All deviations are minor and well-justified. The pipeline reproduces Wong's *methodology* rather than copying Wong's *code* (which was Keras; ours is PyTorch).

---

## What these models can and cannot do

**Can do:**
- Classify a 2D MRI slice into one of 4 tumor types (~94% 5-fold test accuracy for both architectures; see [Phase 1](../phases/phase-1-vgg16-baseline.md) and [Phase 2](../phases/phase-2-architectures.md))
- Produce class probabilities (via softmax on logits) suitable for Grad-CAM, LIME, and SHAP (Phase 3)
- Serve as the two endpoints of the multi-architecture comparison

**Cannot do:**
- Segment the tumor (where it is); only classify (what it is)
- Process 3D volumes; works on individual 2D slices
- Handle MRI scans with classes outside the training distribution (the model will confidently mispredict OOD inputs — a known XAI motivation)
- Replace a radiologist; intended as a research baseline, not a clinical tool

---

## References

- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv:1409.1556.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. arXiv:1512.03385.
- Wong, Y. C., Choi, L. K., Singh Sandhu, R., Mohd Faizal Bin Abdullah, A., Yusuf, R., Saiful Bahari Bin Shaari, A., & Mohamad Shariff, S. F. F. (2025). *Classifying brain tumors on magnetic resonance imaging by deep learning techniques*. PLOS ONE 20(5): e0322624.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are features in deep neural networks?* NIPS.
- Howard, J., & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text Classification*. ACL. (cross-domain transfer-learning principles)
