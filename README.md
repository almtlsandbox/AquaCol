# 🌊 AquaCol — Underwater Image Enhancement

> **Python implementation of the Red Channel Prior paper**, with a full-featured interactive GUI featuring split-view before/after comparison, real-time preview, pan & zoom.

---

## Overview

Underwater photos suffer from a characteristic blue-green colour cast and loss of contrast caused by wavelength-dependent light absorption in water. Red light disappears within the first few metres; green follows; blue travels furthest.

**AquaCol** restores underwater images using the **Red Channel Prior (RCP)** — the observation that, in most local image patches, the red channel contains at least one near-zero pixel. This mirrors the well-known Dark Channel Prior used for haze removal, but adapted to the underwater domain.

The degradation model is:

$$I^c(x) = J^c(x) \cdot t^c(x) + B^c \cdot (1 - t^c(x))$$

| Symbol | Meaning |
|--------|---------|
| $I^c$  | Observed (degraded) image |
| $J^c$  | Latent (restored) scene radiance |
| $t^c$  | Transmission map — fraction of scene light reaching the camera |
| $B^c$  | Background / water-body light |
| $c$    | Colour channel ∈ {R, G, B} |

---

## Features

### Algorithm (`underwater_enhancement.py`)

| Step | Description |
|------|-------------|
| **Red channel compensation** | Pre-boosts the faded red channel before prior estimation |
| **Red Channel Prior** | Patch-wise minimum of the red channel gives the initial transmission map |
| **Background light estimation** | Brightest pixels in the prior locate the water-body illuminant $B$ |
| **Transmission map** | Derived from the normalised red channel with configurable $\omega$ |
| **Guided filter** | Edge-preserving smoothing preserves object boundaries in the map |
| **Per-channel Beer-Lambert** | Each channel's transmission is computed via $t_c = t_R^{\eta_c / \eta_R}$, matching real water absorption spectra |
| **Scene recovery** | Algebraic inversion of the degradation model, clipped at $t_\mathrm{min}$ |
| **Gray-world white balance** | Removes residual colour cast; gain capped at 1.8× to prevent over-saturation |
| **NL-means denoise** | `cv2.fastNlMeansDenoisingColored` suppresses amplified noise in dark regions |
| **CLAHE** | Per-channel adaptive contrast enhancement (clip limit 1.5) |

Three methods are available:

- **`red_channel`** *(recommended)* — full RCP pipeline as above  
- **`dark_channel`** — classic Dark Channel Prior of He et al. (baseline)  
- **`inversion`** — Galdran 2015: image complement + standard dehazing

#### Water type presets

The Beer-Lambert channel ratios vary with water clarity:

| Preset | Green ratio | Blue ratio | Typical scene |
|--------|-------------|------------|---------------|
| `ocean` | 0.25 | 0.12 | Clear blue water, > 10 m visibility |
| `coastal` *(default)* | 0.35 | 0.25 | Moderate turbidity |
| `turbid` | 0.50 | 0.40 | Harbour, pool, murky water |
| `green_water` | 0.60 | 0.55 | Algae-rich / green-dominant |

---

### GUI (`underwater_enhancement_gui.py`)

![GUI layout](docs/gui_preview.png)

#### Split-view canvas
- Drag the **vertical divider** to reveal more *BEFORE* or *AFTER*
- The handle shows left/right arrow triangles for discoverability

#### Pan & Zoom
- **Mouse wheel** — zoom towards the cursor (4 % → 2000 %)
- **Left-drag** — pan anywhere on the canvas
- **Double-click** — fit image to window and reset zoom
- Zoom percentage badge always visible in the bottom-right corner

#### Rotate
- **⟳ Rotate 90°** button — non-destructive; applies to both the main canvas and the transmission map

#### Fast preview
- While moving sliders the image is enhanced on a **downscaled copy** (longest edge ≤ 900 px) for near-instant feedback
- **▶▶ Apply Full Resolution** processes the original file at full quality
- Saving while in preview mode prompts the user to run full-res first

#### Progress indicator
- Full-width animated amber bar appears **below the toolbar** while processing
- Bouncing block with bright centre highlight; hides automatically when done

---

## Installation

```bash
# Clone
git clone https://github.com/almtlsandbox/AquaCol.git
cd AquaCol

# Install dependencies
pip install -r requirements.txt
```

**Requirements** (Python 3.9+):

```
numpy>=1.24
opencv-python>=4.8
scipy>=1.11
matplotlib>=3.7
Pillow>=10.0
```

---

## Usage

### GUI

```bash
python underwater_enhancement_gui.py
```

1. **Load Image** — open any JPG / PNG / TIFF underwater photo
2. Choose **Method** and **Water Type** (or click *Auto-detect*)
3. Adjust sliders — preview updates automatically after 500 ms
4. Click **▶▶ Apply Full Resolution** for the final result
5. **Save Result**

### Command line

```bash
# Basic enhancement
python underwater_enhancement.py path/to/image.jpg

# Specify method and water type
python underwater_enhancement.py img.jpg --method red_channel --water-type ocean

# Auto-detect water type first
python underwater_enhancement.py img.jpg --diagnose

# Run on a built-in synthetic test image
python underwater_enhancement.py --demo --save-fig

# All options
python underwater_enhancement.py --help
```

### Python API

```python
from underwater_enhancement import enhance_underwater_image

results = enhance_underwater_image(
    "my_dive.jpg",
    method      = "red_channel",   # or "dark_channel" / "inversion"
    water_type  = "coastal",       # ocean / coastal / turbid / green_water
    omega       = 0.95,            # correction strength
    t_min       = 0.20,            # minimum transmission (noise floor)
    patch_size  = 15,
)

import cv2
cv2.imwrite("enhanced.jpg", results["enhanced"])
```

Key entries in the returned `dict`:

| Key | Type | Description |
|-----|------|-------------|
| `enhanced` | `ndarray` uint8 | Final enhanced image |
| `original` | `ndarray` uint8 | Input image (copy) |
| `transmission` | `ndarray` float32 | Refined transmission map |
| `background_light` | `ndarray` shape (3,) | Estimated water-body light B |

---

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `red_channel` | Enhancement method |
| `--water-type` | `coastal` | Water type preset |
| `--omega` | `0.95` | Correction strength ω |
| `--t-min` | `0.20` | Minimum transmission t_min |
| `--patch-size` | `15` | Prior patch size (pixels) |
| `--no-guided-filter` | — | Skip guided filter step |
| `--no-white-balance` | — | Skip white balance |
| `--no-denoise` | — | Skip NL-means denoise |
| `--no-clahe` | — | Skip CLAHE step |
| `--diagnose` | — | Print channel stats & recommended water type |
| `--demo` | — | Run on built-in synthetic test image |
| `--save-fig` | — | Save comparison figure to disk |

---

## References

- **He, K., Sun, J., & Tang, X. (2011).** *Single Image Haze Removal Using Dark Channel Prior.* IEEE TPAMI.
- **Galdran, A. et al. (2015).** *Automatic Red-Channel Underwater Image Restoration.* Journal of Visual Communication and Image Representation.
- **Underwater Image Enhancement Based on Red Channel Prior** — the paper this implementation follows.

---

## License

MIT — see `LICENSE` for details.
