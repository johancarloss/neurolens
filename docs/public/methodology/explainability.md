# Explainability — How Grad-CAM turns a decision into a heatmap

> Reference document — explains the **mechanism** of the explainability (XAI)
> techniques used in [Phase 3](../phases/phase-3-xai.md). It assumes the CNN
> internals described in [`model.md`](model.md) (convolution, feature maps,
> backbone/head). **Grad-CAM** is documented here first; **LIME** and **SHAP**
> join as sibling sections as each is written up.

---

## The question XAI answers

A trained model gives a confident answer — *"notumor: 71%"* — but classification
alone hides the reasoning. Two failures look identical from the outside: a model
that looked at the tumor and a model that looked at an unrelated structure can
both output the same label. For a medical baseline that is not good enough; we
need to see **where** the model looked and decide whether to trust it.

**Grad-CAM** (Gradient-weighted Class Activation Mapping) answers exactly that:
it produces a heatmap over the input MRI showing which regions most drove the
predicted class. This document explains how it does so, step by step, and ties
each step to the code that implements it.

---

## The premise: feature maps are what Grad-CAM reads

Grad-CAM does not look at the raw image. It reads the **last convolutional
layer** of the backbone. From [`model.md`](model.md#how-convolution-and-pooling-actually-work),
recall that a convolution filter slides across its input and, at each position,
multiplies-and-sums to produce one number; sliding it everywhere produces a
**feature map** marking where that filter's pattern occurs. A convolutional
layer runs many filters in parallel, so its output is a **stack of feature
maps** — for VGG16's last block, **512 maps of 7×7**; for ResNet50's `layer4`,
**2048 maps of 7×7**.

![A convolution filter sliding over a grayscale image, computing one output number per position, forming an activation (feature) map. The math panel shows the 3×3 window multiplied by the filter weights and summed; the right panel is the feature map filling in.](../assets/convolution-feature-map.png)

*One feature map being built: the filter (cyan box) slides over the image; each
position's multiply-and-sum lands one value in the activation map on the right.
Grad-CAM reads the **stack** of these maps from the last conv layer — the last
place in the network that still knows both **what** was detected and **where**.*

This dual property is the whole reason Grad-CAM targets that specific layer:

| Where you read | Knows **what**? (concept) | Knows **where**? (position) |
|----------------|---------------------------|-----------------------------|
| First conv layer | ✗ only raw edges, no concept | ✓ exact position |
| **Last conv layer** | ✓ "tumor-like texture" | ✓ "…here in this 7×7 region" |
| Final decision (post-head) | ✓ "notumor: 71%" | ✗ position discarded |

The head flattens (VGG16) or globally averages (ResNet50) the feature maps to
decide, which throws away location — see
[`model.md`](model.md#global-average-pooling-and-why-the-head-is-small). The last
conv layer is therefore the last point that retains a spatial grid, and Grad-CAM
exploits it.

---

## How Grad-CAM works — the six-step recipe

```
1. FORWARD    run the image through the backbone; keep the last layer's
              feature maps  A₁ … A₅₁₂   (each 7×7)                 [model.md: forward]

2. BACKWARD   for the target class score, compute the gradient w.r.t. each
              map, then average it → one importance weight αₖ per map

3. COMBINE    weighted sum of the maps:  L = Σₖ  αₖ · Aₖ            [weights, model.md]

4. ReLU       L = max(0, L)  — keep only evidence FOR the class    [ReLU, model.md]

5. UPSAMPLE   stretch the 7×7 result up to 224×224

6. OVERLAY    blend the heatmap over the original MRI (warm = high)
```

**Steps 1 and 2 are the two halves of the name.** The *forward* pass yields the
maps — *what and where*. The *backward* pass answers *which of those maps
mattered for this specific image*: the gradient of the class score with respect
to a feature map is that map's sensitivity — nudge the map up, does the class
score rise (positive, evidence for), fall (negative, evidence against), or stay
put (zero, irrelevant)? Global-average-pooling that gradient gives a single
importance weight `αₖ`. It behaves like the learned weights from
[`model.md`](model.md#how-convolution-and-pooling-actually-work), but computed
**per image** — which is why two different scans produce different heatmaps.

### A worked example with three maps

Suppose the backbone produced just three feature maps `A`, `B`, `C`, and the
backward pass returned these importance weights:

| Map | Weight `αₖ` | Reading |
|-----|-------------|---------|
| `A` | **+3** | strong evidence **for** the class |
| `B` | **0** | did not matter for this image |
| `C` | **−2** | evidence **against** the class |

Step 3 (combine) is literally:

```
L  =  3·A  +  0·B  +  (−2)·C
      └──┘    └──┘    └────┘
   A lights   B drops   C pushes
   up fully   out       negative
```

Step 4 (ReLU) then clips everything `C` pushed below zero back to zero, so the
final heatmap is essentially **where map `A` was active**. Because `A` is a 7×7
grid that still encodes position, the hot region lands on the exact part of the
image that drove the decision. Evidence *against* the class (`C`) never lights up
— a direct consequence of the ReLU in step 4, and the reason Grad-CAM only ever
shows support *for* the explained class.

### Why the heatmap looks coarse

Step 5 stretches a **7×7** map to **224×224** — a 32× upscale — so Grad-CAM
output is a smooth blob, not a pixel-sharp outline. This is a fundamental
resolution limit of the technique (inherited from the last layer's spatial size,
see [`model.md`](model.md#why-spatial-size-shrinks-while-depth-grows)), and one
reason Phase 3 cross-checks Grad-CAM against LIME and SHAP rather than trusting
any single map.

---

## In code

The wrapper lives in
[`src/neurolens/xai/gradcam.py`](../../../src/neurolens/xai/gradcam.py). It
delegates steps 1–5 to the [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam)
library and supplies the architecture-correct target layer:

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from neurolens.models.factory import get_target_layer_for_gradcam

class GradCAMExplainer:
    def __init__(self, model, arch, device):
        self.model = model.eval().to(device)
        self.target_layer = get_target_layer_for_gradcam(model, arch)  # last conv block

    def explain(self, input_tensor, target_class):
        targets = [ClassifierOutputTarget(target_class)]               # "explain THIS class"
        with GradCAM(model=self.model, target_layers=[self.target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # steps 1–5
        return grayscale_cam, elapsed_ms
```

Two design points make this architecture-agnostic:

- **The target layer is resolved by the factory.**
  [`get_target_layer_for_gradcam(model, arch)`](../../../src/neurolens/models/factory.py)
  returns `features[-1]` for VGG16 and `layer4[-1]` for ResNet50 — the "last conv
  layer" from the table above, per architecture, with no branching in the
  explainer.
- **The target class is explicit.** `ClassifierOutputTarget(target_class)` is
  what makes the backward pass (step 2) start from the chosen class's score. The
  demo passes the model's own top prediction, so the heatmap explains *the
  decision the model actually made*.

The demo composes this with the overlay (step 6) in
[`src/neurolens/ui/inference.py`](../../../src/neurolens/ui/inference.py): it runs
`explain(...)` for both architectures and blends each map over the shared
original MRI with `overlay_saliency`, so the two models' attention can be
compared side by side.

---

## What Grad-CAM revealed in this project

Grad-CAM is not a decoration — in Phase 3 it surfaced a real failure mode
(Finding 8). On a glioma that **both** models misread as *notumor*, the heatmaps
did not spread out from uncertainty; they concentrated **confidently on the
ventricles** (center) while ignoring the tumor in the frontal lobe. Reading the
recipe backward explains exactly what happened: the forward pass activated a
"suspicious texture" map over the ventricles, the backward pass gave that map a
high weight for the *notumor* score, and combine → ReLU → upsample painted the
ventricles hot. The model had learned *"ventricular distortion ⇒ tumor"* and
looks there by default.

![Grad-CAM, LIME and SHAP overlays for both architectures on a glioma misread as notumor, each shown beside the original MRI. Grad-CAM concentrates on the ventricles, not the frontal-lobe tumor.](../assets/phase-3-case-error.png)

*The error case from [Phase 3](../phases/phase-3-xai.md): with the original MRI
beside each overlay, "a large red blob" resolves into "the model looked in the
wrong place" — which is only legible because the last conv layer preserved
**where**.*

---

## Limitations

- **Coarse resolution.** The 7×7 → 224×224 upscale makes Grad-CAM a
  region-level, not pixel-level, explanation.
- **Positive evidence only.** The ReLU in step 4 discards evidence *against* the
  class, so a Grad-CAM map shows only what supported the chosen label.
- **Attention is not correctness.** A confident, well-placed heatmap shows
  *where* the model looked, not *whether the reasoning generalizes*. Phase 3's
  Finding 10 (shortcut learning on the skull, surfaced by LIME) is the
  cautionary counterpart — which is why the project reports **three** techniques,
  not one.

---

## Next

LIME and SHAP are documented as sibling sections here as each is written up. In
short: **LIME** perturbs the image (hides superpixels) and watches the decision
change; **SHAP** attributes a fair, game-theoretic contribution to each pixel.
Both answer the same question as Grad-CAM — *where did the evidence come from?* —
from independent angles, which is what makes their agreement (or disagreement) in
Phase 3 meaningful.

---

## References

- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
  (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
  Localization*. ICCV. arXiv:1610.02391.
- Gildenblat, J. et al. *pytorch-grad-cam* (library). https://github.com/jacobgil/pytorch-grad-cam
