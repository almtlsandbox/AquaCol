"""
Underwater Image Enhancement Based on Red Channel Prior
=======================================================

Physical background
-------------------
Water selectively absorbs light according to wavelength.  Red light is
absorbed within the first few metres, then green, and blue last. This
produces the characteristic blue-green cast of underwater images and the
very low red-channel signal that motivates the *Red Channel Prior* (RCP).

The degradation model (single-channel) is:

    I^c(x)  =  J^c(x) · t^c(x)  +  B^c · (1 − t^c(x))

where
    I   – observed (degraded) image
    J   – latent (restored) scene radiance
    t   – transmission map  (fraction of scene light reaching the camera)
    B   – background / water light
    c ∈ {R, G, B}

The Red Channel Prior states that, in most local patches of an underwater
image, at least one pixel in the *red* channel has a very small value,
analogous to the Dark Channel Prior of He et al. used for dehazing.

Pipeline implemented
--------------------
1.  Red channel compensation   – pre-boosts the red channel before the prior
    is applied, so that the estimated transmission is more accurate.
2.  Red Channel Prior          – minimum red value in each local patch gives
    an initial estimate of the transmission map.
3.  Background light estimation – brightest pixels in the prior map are used
    to locate and estimate the water-body light B.
4.  Transmission map estimation – derived from the normalised red channel.
5.  Guided filter refinement   – edge-preserving smoothing of the raw map.
6.  Scene recovery             – algebraic inversion of the degradation model.
7.  Gray-world white balance   – removes residual colour cast.
8.  CLAHE                      – per-channel adaptive contrast enhancement.

An alternative *inversion-based* method (Galdran et al., 2015) is also
provided: complementing the image converts the underwater problem into a
standard dehazing problem solvable with the Dark Channel Prior.

Usage
-----
    # From the command line:
    python underwater_enhancement.py path/to/image.jpg

    # Demo with synthetic underwater image:
    python underwater_enhancement.py --demo

    # Choose a different method:
    python underwater_enhancement.py img.jpg --method inversion

    # From Python:
    from underwater_enhancement import enhance_underwater_image
    results = enhance_underwater_image("my_image.jpg")
    cv2.imwrite("out.png", results["enhanced"])
"""

from __future__ import annotations

import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter


# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------

def dark_channel(img_float: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Compute the Dark Channel of *img_float*.

    For every pixel the dark channel value is the minimum intensity across
    all three colour channels within a square patch of side *patch_size*.

    Parameters
    ----------
    img_float : np.ndarray, shape (H, W, 3), float64 in [0, 1], BGR
    patch_size : int – side length of the local patch

    Returns
    -------
    np.ndarray, shape (H, W), float64 in [0, 1]
    """
    min_channel = np.min(img_float, axis=2)               # pixel-wise channel min
    dark = minimum_filter(min_channel, size=patch_size)   # spatial min in patch
    return dark


def red_channel_prior(img_float: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Compute the Red Channel Prior (RCP) map.

    The RCP is the spatial minimum of the **red** channel value within every
    local patch.  Because water absorbs red light most aggressively, these
    local minima are usually very small in genuine underwater images.

    Parameters
    ----------
    img_float : np.ndarray, shape (H, W, 3), float64 in [0, 1], BGR
    patch_size : int

    Returns
    -------
    np.ndarray, shape (H, W), float64 in [0, 1]
    """
    red = img_float[:, :, 2]                               # index 2 = Red in BGR
    return minimum_filter(red, size=patch_size)


def estimate_background_light(
    img_float: np.ndarray,
    prior: np.ndarray,
    top_fraction: float = 0.001,
) -> np.ndarray:
    """
    Estimate the background (water body) light **B**.

    The brightest pixels in the prior map are assumed to correspond to
    regions where the transmission is smallest (deepest water / most haze).
    Among those candidate pixels the one with the highest overall brightness
    is selected as the background-light colour.

    Parameters
    ----------
    img_float    : np.ndarray (H, W, 3) float in [0, 1]
    prior        : np.ndarray (H, W) – dark or red channel prior
    top_fraction : float – fraction of brightest prior pixels to consider

    Returns
    -------
    np.ndarray, shape (3,), float in [0, 1]  –  B^B, B^G, B^R
    """
    h, w = prior.shape
    n_candidates = max(1, int(h * w * top_fraction))

    flat_prior = prior.ravel()
    flat_img   = img_float.reshape(-1, 3)

    # Indices of the top-n brightest pixels in the prior
    indices = np.argpartition(flat_prior, -n_candidates)[-n_candidates:]

    # Among candidates pick the one with the highest L2 norm across channels
    candidates = flat_img[indices]
    best = np.argmax(np.linalg.norm(candidates, axis=1))
    return candidates[best].copy()


def estimate_transmission(
    img_float: np.ndarray,
    background_light: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
) -> np.ndarray:
    """
    Estimate the transmission map **t** using the Red Channel Prior.

    After normalising the image by the background light (so that the
    background maps to 1), the minimum red value in each patch gives a
    proxy for how much of the water-light has been mixed in.

        t(x) = 1 − ω · min_{y ∈ Ω(x)}( I_R(y) / B_R )

    The factor *omega* < 1 intentionally keeps a small residual of the water
    colour so that the final image does not look over-processed.

    Parameters
    ----------
    img_float        : (H, W, 3) float in [0, 1]
    background_light : (3,) float in [0, 1]
    patch_size       : int
    omega            : float in (0, 1]

    Returns
    -------
    np.ndarray (H, W), float in [0, 1]
    """
    # Normalise every channel by the corresponding background-light component
    img_norm = img_float / (background_light + 1e-8)

    # Red-channel prior on the normalised image
    red_norm  = img_norm[:, :, 2]
    red_min   = minimum_filter(red_norm, size=patch_size)

    transmission = 1.0 - omega * red_min
    return np.clip(transmission, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-channel transmission (Beer-Lambert)
# ---------------------------------------------------------------------------

# Water-type presets: (η_G/η_R,  η_B/η_R)
# A smaller ratio means the channel is less attenuated than red, so its
# transmission will be *higher* (less correction needed).
_WATER_TYPE_RATIOS = {
    #                  green ratio   blue ratio
    "ocean":          (0.25,         0.12),   # clear open ocean – very blue
    "coastal":        (0.35,         0.25),   # moderate turbidity (default)
    "turbid":         (0.50,         0.40),   # high turbidity, pool, harbour
    "green_water":    (0.60,         0.55),   # green-dominant water
}


def per_channel_transmission(
    t_red: np.ndarray,
    green_ratio: float = 0.35,
    blue_ratio:  float = 0.25,
) -> np.ndarray:
    """
    Derive per-channel transmission maps from the red-channel estimate.

    Based on Beer-Lambert law each channel's transmission decays
    exponentially with depth at a rate proportional to its attenuation
    coefficient η_c::

        t_c(x) = exp(−η_c · d(x))

    Since ``t_R = exp(−η_R · d)`` we have:

        t_c(x) = t_R(x)^(η_c / η_R)

    Parameters
    ----------
    t_red        : (H, W) float in [0, 1]  – red channel transmission
    green_ratio  : float – η_G / η_R  (green attenuation relative to red)
    blue_ratio   : float – η_B / η_R  (blue  attenuation relative to red)

    Returns
    -------
    np.ndarray (H, W, 3) float in [0, 1],  channels in BGR order
    """
    t_safe = np.clip(t_red, 1e-6, 1.0)           # avoid log(0)
    t_G = np.power(t_safe, green_ratio)           # less attenuated → higher t
    t_B = np.power(t_safe, blue_ratio)            # least attenuated → highest t
    return np.stack([t_B, t_G, t_red], axis=2)   # BGR


def diagnose_channels(img_input) -> dict:
    """
    Print and return per-channel statistics to verify that the image actually
    follows the Red Channel Prior assumptions.

    A genuine underwater image should satisfy:
      mean_R  <  mean_G  <  mean_B   (red is most attenuated)

    Parameters
    ----------
    img_input : str or np.ndarray (uint8, BGR)

    Returns
    -------
    dict with keys ``mean_B``, ``mean_G``, ``mean_R``,
    ``red_is_lowest``, ``recommended_water_type``
    """
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_input}")
    else:
        img = img_input

    f = img.astype(np.float64) / 255.0
    mb, mg, mr = f[:, :, 0].mean(), f[:, :, 1].mean(), f[:, :, 2].mean()

    print("\n── Channel Diagnostics ────────────────────────────────────")
    print(f"  Mean Blue  (index 0): {mb:.4f}")
    print(f"  Mean Green (index 1): {mg:.4f}")
    print(f"  Mean Red   (index 2): {mr:.4f}")

    red_is_lowest = (mr < mg) and (mr < mb)
    if red_is_lowest:
        print("  ✓ Red is the lowest channel — RCP assumption holds.")
    else:
        dominant = ["Blue", "Green", "Red"][int(np.argmin([mb, mg, mr]))]
        print(f"  ✗ Red is NOT the lowest channel (lowest = {dominant}).")
        print("    The Red Channel Prior may not be suitable for this image.")
        print("    Consider --method inversion or --no-white-balance.")

    # Suggest water type based on blue/red ratio
    ratio = float(mb / (mr + 1e-8))
    if ratio > 2.5:
        rec = "ocean"
    elif ratio > 1.5:
        rec = "coastal"
    elif ratio > 1.1:
        rec = "turbid"
    else:
        rec = "green_water"
    print(f"  Blue/Red ratio: {ratio:.2f}  → recommended --water-type {rec}")
    print("────────────────────────────────────────────────────────────\n")

    return {
        "mean_B": mb, "mean_G": mg, "mean_R": mr,
        "red_is_lowest": red_is_lowest,
        "recommended_water_type": rec,
    }


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 60,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """
    Guided Image Filter (He et al., 2013) for edge-preserving refinement.

    Smooths *src* while preserving the edges encoded in *guide*.  Applied
    here to remove the block artefacts introduced by patch-wise transmission
    estimation while keeping boundaries crisp.

    Parameters
    ----------
    guide   : (H, W) or (H, W, 3) float in [0, 1]  – guide image
    src     : (H, W) float in [0, 1]                – image to filter
    radius  : int   – half-window size (full window = 2·radius+1)
    epsilon : float – regularisation that controls smoothing strength

    Returns
    -------
    np.ndarray (H, W) float in [0, 1]
    """
    # Convert guide to grayscale luminance if it is colour
    if guide.ndim == 3:
        # Standard luminance weights (BT-601): R=0.2989 G=0.5870 B=0.1140
        # OpenCV is BGR so: B[0], G[1], R[2]
        I = (0.1140 * guide[:, :, 0]
             + 0.5870 * guide[:, :, 1]
             + 0.2989 * guide[:, :, 2]).astype(np.float64)
    else:
        I = guide.astype(np.float64)

    p = src.astype(np.float64)
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I  = cv2.boxFilter(I,     ddepth=-1, ksize=ksize)
    mean_p  = cv2.boxFilter(p,     ddepth=-1, ksize=ksize)
    mean_Ip = cv2.boxFilter(I * p, ddepth=-1, ksize=ksize)
    mean_II = cv2.boxFilter(I * I, ddepth=-1, ksize=ksize)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I  = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + epsilon)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=ksize)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=ksize)

    q = mean_a * I + mean_b
    return np.clip(q, 0.0, 1.0).astype(np.float64)


def scene_recovery(
    img_float: np.ndarray,
    transmission: np.ndarray,
    background_light: np.ndarray,
    t_min: float = 0.1,
) -> np.ndarray:
    """
    Recover the scene radiance **J** by inverting the degradation model.

        J^c(x) = ( I^c(x) − B^c ) / max(t^c(x), t_min)  +  B^c

    The clamp *t_min* prevents division by very small transmission values
    which would cause noise amplification in dense-water regions.

    Parameters
    ----------
    img_float        : (H, W, 3) float in [0, 1]
    transmission     : (H, W) or (H, W, 3) float in [0, 1].
                       If (H, W), the same map is broadcast to all channels
                       (suitable for the inversion method).
                       If (H, W, 3), a separate map is used per channel
                       (physically correct for the red-channel method).
    background_light : (3,)      float in [0, 1]
    t_min            : float – minimum allowed transmission

    Returns
    -------
    np.ndarray (H, W, 3) float in [0, 1]
    """
    if transmission.ndim == 2:
        t = np.maximum(transmission[:, :, np.newaxis], t_min)   # (H, W, 1) broadcast
    else:
        t = np.maximum(transmission, t_min)                     # (H, W, 3) per-channel
    J = (img_float - background_light) / t + background_light
    return np.clip(J, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Red-channel-specific pre/post processing
# ---------------------------------------------------------------------------

def compensate_red_channel(img_float: np.ndarray) -> np.ndarray:
    """
    Pre-compensate the red (and blue) channel attenuation.

    Before applying the Red Channel Prior, this step pre-boosts the most
    attenuated channels.  The compensation formula mirrors the physics:

        R_comp = R + (mean_G − mean_R) · (1 − R) · G

    A pixel that is already bright (R≈1 or G≈0) receives no compensation,
    while dark-red regions with strong green signal are boosted most.
    The same formula is applied to the blue channel as a secondary
    correction (blue is less attenuated than red but more than green in
    most ocean/pool water types).

    Parameters
    ----------
    img_float : (H, W, 3) float in [0, 1], BGR

    Returns
    -------
    np.ndarray (H, W, 3) float in [0, 1], BGR
    """
    b, g, r = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))

    # Boost red channel
    r_comp = r + (mean_g - mean_r) * (1.0 - r) * g

    # Moderate boost to blue channel
    b_comp = b + (mean_g - mean_b) * (1.0 - b) * g

    result = np.stack([b_comp, g, r_comp], axis=2)
    return np.clip(result, 0.0, 1.0)


def white_balance_gray_world(
    img_float: np.ndarray,
    max_gain: float = 1.8,
) -> np.ndarray:
    """
    Gray-world white balance with a per-channel gain cap.

    Scales each channel so that its mean equals the overall mean.  The
    *max_gain* cap prevents any single channel from being boosted more than
    that factor – without it, a very-low red channel (typical in underwater
    images) would be amplified aggressively, turning the water background
    from blue to teal/orange.

    Parameters
    ----------
    img_float : (H, W, 3) float in [0, 1]
    max_gain  : float – maximum per-channel multiplier (default 1.8)

    Returns
    -------
    np.ndarray (H, W, 3) float in [0, 1]
    """
    means = img_float.reshape(-1, 3).mean(axis=0)      # (B_mean, G_mean, R_mean)
    overall = means.mean()
    scale = overall / (means + 1e-8)                   # per-channel scale factor
    scale = np.clip(scale, 0.0, max_gain)              # cap to avoid over-correction
    return np.clip(img_float * scale, 0.0, 1.0)


def denoise_image(img_uint8: np.ndarray, strength: int = 7) -> np.ndarray:
    """
    Fast non-local means denoising (colour).

    Applied *after* scene recovery and *before* CLAHE to suppress the
    noise amplification that occurs in low-transmission (dark) regions.
    Lower ``strength`` preserves more fine texture; higher values give a
    cleaner but softer result.

    Parameters
    ----------
    img_uint8 : (H, W, 3) uint8 BGR
    strength  : int – filter strength h (same value used for luminance and
                colour components)

    Returns
    -------
    np.ndarray (H, W, 3) uint8
    """
    return cv2.fastNlMeansDenoisingColored(
        img_uint8,
        None,
        h=strength,
        hColor=strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def apply_clahe(
    img_uint8: np.ndarray,
    clip_limit: float = 1.5,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation (CLAHE), per channel.

    CLAHE divides the image into small tiles and equalises the histogram of
    each tile independently, then interpolates between them.  The clip limit
    controls how aggressively the histogram is redistributed, preventing
    noise over-amplification.

    Parameters
    ----------
    img_uint8       : (H, W, 3) uint8 in [0, 255], BGR
    clip_limit      : float
    tile_grid_size  : (int, int)

    Returns
    -------
    np.ndarray (H, W, 3) uint8, BGR
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    b, g, r = cv2.split(img_uint8)
    return cv2.merge([clahe.apply(b), clahe.apply(g), clahe.apply(r)])


# ---------------------------------------------------------------------------
# Alternative: Inversion-based method (Galdran et al. 2015)
# ---------------------------------------------------------------------------

def inversion_based_enhancement(
    img_float: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t_min: float = 0.1,
) -> tuple:
    """
    Inversion-based underwater enhancement (Galdran et al., 2015).

    Key insight: complementing an underwater image (1 − I) produces an
    image that *looks* like a hazy outdoor image.  Applying the standard
    Dark Channel Prior to the complemented image therefore models the
    red channel attenuation directly.

    Steps
    -----
    1. Invert : I_inv = 1 − I
    2. Dark channel of I_inv
    3. Estimate background light of I_inv
    4. Transmission from dark channel of normalised I_inv
    5. Guided-filter refinement
    6. Scene recovery in the inverted domain
    7. Re-invert: J = 1 − J_inv

    Parameters
    ----------
    img_float : (H, W, 3) float in [0, 1], BGR
    patch_size, omega, t_min : see ``enhance_underwater_image``

    Returns
    -------
    enhanced   : np.ndarray (H, W, 3) float in [0, 1]
    transmission : np.ndarray (H, W) float in [0, 1]
    background_light : np.ndarray (3,) float in [0, 1]
    """
    img_inv = 1.0 - img_float

    dark = dark_channel(img_inv, patch_size)
    bg   = estimate_background_light(img_inv, dark)

    # Normalise and recompute dark channel for transmission
    img_inv_norm = img_inv / (bg + 1e-8)
    dark_norm    = dark_channel(img_inv_norm, patch_size)
    trans        = np.clip(1.0 - omega * dark_norm, 0.0, 1.0)

    # Guided-filter refinement guided by the *original* image luminance
    guide_gray = (0.1140 * img_float[:, :, 0]
                  + 0.5870 * img_float[:, :, 1]
                  + 0.2989 * img_float[:, :, 2])
    trans_refined = guided_filter(guide_gray, trans, radius=60, epsilon=1e-3)

    J_inv = scene_recovery(img_inv, trans_refined, bg, t_min)
    J     = np.clip(1.0 - J_inv, 0.0, 1.0)
    return J, trans_refined, bg


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def enhance_underwater_image(
    img_input,
    patch_size: int = 15,
    omega: float = 0.95,
    t_min: float = 0.2,
    use_guided_filter: bool = True,
    apply_white_balance: bool = True,
    apply_denoise: bool = True,
    apply_clahe_post: bool = True,
    method: str = "red_channel",
    water_type: str = "coastal",
) -> dict:
    """
    Enhance an underwater image using the Red Channel Prior pipeline.

    Parameters
    ----------
    img_input : str or np.ndarray
        Path to an image file **or** a BGR numpy array (uint8 or float [0,1]).
    patch_size : int
        Side length of the local patch used to compute channel priors.
        Larger values produce smoother but less spatially accurate maps.
    omega : float in (0, 1]
        Haze/fog-retention factor.  Values close to 1 produce the most vivid
        restoration; lower values retain more of the original water colour.
    t_min : float in (0, 1)
        Minimum allowed transmission.  Higher values reduce noise amplification
        in dark (low-transmission) regions.  Default 0.2 gives a good
        trade-off; use 0.1 only for clear water with gentle foregrounds.
    use_guided_filter : bool
        If True, refine the raw transmission map with the guided filter.
    apply_white_balance : bool
        If True, apply gray-world white balance (gain capped at 1.8×) after
        scene recovery.
    apply_denoise : bool
        If True, apply fast NL-means denoising after scene recovery and
        before CLAHE.  Suppresses noise amplification in dark regions.
    apply_clahe_post : bool
        If True, apply per-channel CLAHE as the final step.
    method : str
        ``"red_channel"``  – Red Channel Prior with per-channel transmission
                             (default, physically correct).
        ``"dark_channel"`` – Classic Dark Channel Prior (He et al.) – useful
                             as a baseline comparison.
        ``"inversion"``    – Inversion-based method (Galdran et al. 2015).
    water_type : str
        Controls the per-channel Beer-Lambert attenuation ratios (η_c/η_R):
        ``"ocean"``       – clear blue water  (G=0.25, B=0.12)
        ``"coastal"``     – moderate turbidity (G=0.35, B=0.25)  [default]
        ``"turbid"``      – high turbidity / pool  (G=0.50, B=0.40)
        ``"green_water"`` – green-dominant water   (G=0.60, B=0.55)
        Ignored for the ``inversion`` method.
        Run with ``--diagnose`` to get a recommendation for your image.

    Returns
    -------
    dict with keys:
        ``"enhanced"``         – Enhanced image, uint8 BGR
        ``"transmission"``     – Red-channel transmission map, float (H, W)
        ``"background_light"`` – Estimated background light, float (3,) BGR
        ``"original"``         – Original image, uint8 BGR
    """
    # ------------------------------------------------------------------
    # Load / validate input
    # ------------------------------------------------------------------
    if isinstance(img_input, str):
        img_uint8 = cv2.imread(img_input)
        if img_uint8 is None:
            raise FileNotFoundError(f"Cannot read image: {img_input}")
    elif img_input.dtype == np.uint8:
        img_uint8 = img_input.copy()
    else:
        img_uint8 = (np.clip(img_input, 0.0, 1.0) * 255.0).astype(np.uint8)

    original  = img_uint8.copy()
    img_float = img_uint8.astype(np.float64) / 255.0

    # ------------------------------------------------------------------
    # Method: inversion-based (Galdran et al.)
    # ------------------------------------------------------------------
    if method == "inversion":
        enhanced_float, transmission, background_light = inversion_based_enhancement(
            img_float, patch_size, omega, t_min
        )

    # ------------------------------------------------------------------
    # Methods: red_channel or dark_channel
    # ------------------------------------------------------------------
    else:
        # Step 1 – Red channel compensation (skip for pure dark-channel baseline)
        if method == "red_channel":
            img_proc = compensate_red_channel(img_float)
        else:
            img_proc = img_float.copy()

        # Step 2 – Compute the channel prior
        if method == "red_channel":
            prior = red_channel_prior(img_proc, patch_size)
        else:
            prior = dark_channel(img_proc, patch_size)

        # Step 3 – Background light estimation
        background_light = estimate_background_light(img_proc, prior)

        # Step 4 – Transmission map (red channel only)
        transmission = estimate_transmission(
            img_proc, background_light, patch_size, omega
        )

        # Step 5 – Guided-filter refinement
        if use_guided_filter:
            guide_gray = (0.1140 * img_proc[:, :, 0]
                          + 0.5870 * img_proc[:, :, 1]
                          + 0.2989 * img_proc[:, :, 2])
            transmission = guided_filter(
                guide_gray, transmission, radius=60, epsilon=1e-3
            )

        # Step 6 – Per-channel transmission (Beer-Lambert correction)
        # A single red-channel transmission applied to all channels would
        # severely over-correct green and blue (which are far less attenuated).
        g_ratio, b_ratio = _WATER_TYPE_RATIOS.get(
            water_type, _WATER_TYPE_RATIOS["coastal"]
        )
        transmission_3ch = per_channel_transmission(
            transmission, green_ratio=g_ratio, blue_ratio=b_ratio
        )

        # Step 7 – Scene recovery with per-channel transmission
        enhanced_float = scene_recovery(
            img_proc, transmission_3ch, background_light, t_min
        )

    # ------------------------------------------------------------------
    # Post-processing (shared by all methods)
    # ------------------------------------------------------------------
    # Step 7 – White balance (gain capped to avoid over-correction)
    if apply_white_balance:
        enhanced_float = white_balance_gray_world(enhanced_float)

    enhanced_uint8 = (enhanced_float * 255.0).astype(np.uint8)

    # Step 8 – Denoise (suppresses noise from low-transmission amplification)
    if apply_denoise:
        enhanced_uint8 = denoise_image(enhanced_uint8)

    # Step 9 – CLAHE
    if apply_clahe_post:
        enhanced_uint8 = apply_clahe(enhanced_uint8)

    return {
        "enhanced":         enhanced_uint8,
        "transmission":     transmission,
        "background_light": background_light,
        "original":         original,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(original: np.ndarray, enhanced: np.ndarray) -> dict:
    """
    Compute objective image quality metrics.

    Metrics returned
    ----------------
    mse                       – Mean Squared Error (pixel scale 0–255)
    psnr                      – Peak Signal-to-Noise Ratio in dB
    original_channel_means    – Per-channel mean intensity (B,G,R) of original
    enhanced_channel_means    – Per-channel mean intensity (B,G,R) of enhanced
    original_colorfulness     – Colorfulness metric (Hasler & Suesstrunk 2003)
    enhanced_colorfulness     – Colorfulness metric of enhanced image
    red_channel_gain          – Ratio of enhanced to original red-channel mean
    """
    def _colorfulness(img: np.ndarray) -> float:
        """Hasler & Suesstrunk (2003) colorfulness metric."""
        b = img[:, :, 0].astype(float)
        g = img[:, :, 1].astype(float)
        r = img[:, :, 2].astype(float)
        rg = r - g
        yb = 0.5 * (r + g) - b
        sigma = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        mu    = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
        return float(sigma + 0.3 * mu)

    o = original.astype(np.float64)
    e = enhanced.astype(np.float64)

    mse  = float(np.mean((o - e) ** 2))
    psnr = float(10.0 * np.log10(255.0 ** 2 / (mse + 1e-10)))

    o_means = original.reshape(-1, 3).mean(axis=0)
    e_means = enhanced.reshape(-1, 3).mean(axis=0)

    red_gain = float(e_means[2] / (o_means[2] + 1e-8))

    return {
        "mse":                    mse,
        "psnr":                   psnr,
        "original_channel_means": o_means,      # B, G, R
        "enhanced_channel_means": e_means,      # B, G, R
        "original_colorfulness":  _colorfulness(original),
        "enhanced_colorfulness":  _colorfulness(enhanced),
        "red_channel_gain":       red_gain,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_results(results: dict, save_path: str | None = None):
    """
    Display (and optionally save) a side-by-side comparison figure plus
    per-channel histograms.

    Parameters
    ----------
    results   : dict returned by ``enhance_underwater_image``
    save_path : str or None – if given, the figure is saved to this path;
                the histogram figure is saved to the same directory with
                ``_histograms`` appended before the extension.
    """
    original     = results["original"]
    enhanced     = results["enhanced"]
    transmission = results["transmission"]

    # ---- Main comparison figure ----------------------------------------
    fig1, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle("Underwater Image Enhancement – Red Channel Prior", fontsize=15)

    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")

    im = axes[1].imshow(transmission, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Transmission Map  t(x)", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Enhanced (Red Channel Prior)", fontsize=13)
    axes[2].axis("off")

    plt.tight_layout()

    # ---- Histogram comparison figure -----------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Channel Histograms", fontsize=13)

    channel_labels = ["Blue", "Green", "Red"]
    channel_colors = ["royalblue", "limegreen", "tomato"]

    for c, (lbl, col) in enumerate(zip(channel_labels, channel_colors)):
        h_orig = cv2.calcHist([original],  [c], None, [256], [0, 256]).ravel()
        h_enh  = cv2.calcHist([enhanced], [c], None, [256], [0, 256]).ravel()
        axes2[0].plot(h_orig, color=col, alpha=0.8, label=lbl)
        axes2[1].plot(h_enh,  color=col, alpha=0.8, label=lbl)

    for ax, title in zip(axes2, ["Original", "Enhanced"]):
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()

    if save_path:
        fig1.savefig(save_path, dpi=150, bbox_inches="tight")
        base, ext = os.path.splitext(save_path)
        hist_path = base + "_histograms" + ext
        fig2.savefig(hist_path, dpi=150, bbox_inches="tight")
        print(f"Figures saved:\n  {save_path}\n  {hist_path}")

    plt.show()
    return fig1, fig2


# ---------------------------------------------------------------------------
# Synthetic underwater image generator (for smoke-testing)
# ---------------------------------------------------------------------------

def make_synthetic_underwater_image(
    height: int = 480,
    width:  int = 640,
    seed:   int = 42,
) -> np.ndarray:
    """
    Generate a synthetic underwater-like BGR image for testing purposes.

    The image is created by:
    - Producing a random base scene with medium brightness
    - Simulating red-channel absorption (strong depth-dependent attenuation)
    - Applying a depth-dependent blue-green veil (backscatter)
    - Adding a slight depth gradient (darker at bottom)
    """
    rng = np.random.default_rng(seed)

    # Base scene (random blobs of varying brightness)
    scene = rng.integers(80, 220, (height, width, 3), dtype=np.uint8)

    # Depth gradient (y-axis → deeper = darker and more blue)
    y_grad = np.linspace(0.0, 1.0, height)[:, np.newaxis]          # (H,1)

    # Simulate channel-selective attenuation
    # Red attenuates most, green intermediate, blue least
    scene_f = scene.astype(np.float32) / 255.0
    scene_f[:, :, 2] *= np.clip(1.0 - 0.85 * y_grad, 0.05, 1.0)   # Red
    scene_f[:, :, 1] *= np.clip(1.0 - 0.40 * y_grad, 0.30, 1.0)   # Green
    scene_f[:, :, 0] *= np.clip(1.0 - 0.15 * y_grad, 0.60, 1.0)   # Blue

    # Add blue-green backscatter veil
    veil_b = 0.30 * y_grad
    veil_g = 0.18 * y_grad
    scene_f[:, :, 0] = np.clip(scene_f[:, :, 0] + veil_b, 0.0, 1.0)
    scene_f[:, :, 1] = np.clip(scene_f[:, :, 1] + veil_g, 0.0, 1.0)

    img = (scene_f * 255.0).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Underwater Image Enhancement Based on Red Channel Prior.\n"
            "Applies red-channel compensation, transmission-map estimation,\n"
            "guided-filter refinement, scene recovery, white balance and CLAHE."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "input", nargs="?",
        help="Path to input underwater image. Omit together with --demo to run demo.",
    )
    p.add_argument("-o", "--output", default=None,
                   help="Output path for the enhanced image.")
    p.add_argument("--method",
                   choices=["red_channel", "dark_channel", "inversion"],
                   default="red_channel",
                   help="Enhancement method.")
    p.add_argument("--patch-size", type=int, default=15,
                   help="Local patch size for the channel prior.")
    p.add_argument("--omega", type=float, default=0.95,
                   help="Haze-retention factor ω ∈ (0, 1].")
    p.add_argument("--t-min", type=float, default=0.2,
                   help="Minimum transmission t_min ∈ (0, 1). Higher = less noise in dark areas.")
    p.add_argument("--water-type",
                   choices=list(_WATER_TYPE_RATIOS.keys()),
                   default="coastal",
                   help="Per-channel attenuation preset (Beer-Lambert ratios). "
                        "Run --diagnose first to find the best value for your image.")
    p.add_argument("--no-guided-filter",   action="store_true",
                   help="Disable guided-filter transmission refinement.")
    p.add_argument("--no-white-balance",   action="store_true",
                   help="Disable gray-world white balance.")
    p.add_argument("--no-denoise",         action="store_true",
                   help="Disable NL-means denoising.")
    p.add_argument("--no-clahe",           action="store_true",
                   help="Disable CLAHE post-processing.")
    p.add_argument("--save-fig",           default="enhancement_result.png",
                   help="File path for the visualisation figure.")
    p.add_argument("--demo",               action="store_true",
                   help="Run on a synthetic underwater image (no real image needed).")
    p.add_argument("--diagnose",           action="store_true",
                   help="Print channel statistics and recommended --water-type, then exit.")
    return p


def main():
    args = _build_parser().parse_args()

    # ------------------------------------------------------------------
    # Diagnose-only mode
    # ------------------------------------------------------------------
    if args.diagnose:
        if not args.input:
            print("[Error] --diagnose requires an input image path.")
            return
        diagnose_channels(args.input)
        return

    # ------------------------------------------------------------------
    # Determine input
    # ------------------------------------------------------------------
    if args.demo or args.input is None:
        print("[Demo] Generating synthetic underwater image …")
        img_input = make_synthetic_underwater_image()
        cv2.imwrite("synthetic_underwater.png", img_input)
        print("  → Saved to synthetic_underwater.png")
    else:
        img_input = args.input

    print(f"[Info] Method      : {args.method}")
    print(f"[Info] Water type  : {args.water_type}")
    print(f"[Info] Patch size  : {args.patch_size}")
    print(f"[Info] Omega (ω)   : {args.omega}")
    print(f"[Info] t_min       : {args.t_min}")

    # ------------------------------------------------------------------
    # Run enhancement
    # ------------------------------------------------------------------
    results = enhance_underwater_image(
        img_input,
        patch_size         = args.patch_size,
        omega              = args.omega,
        t_min              = args.t_min,
        use_guided_filter  = not args.no_guided_filter,
        apply_white_balance= not args.no_white_balance,
        apply_denoise      = not args.no_denoise,
        water_type         = args.water_type,
        apply_clahe_post   = not args.no_clahe,
        method             = args.method,
    )

    # ------------------------------------------------------------------
    # Save enhanced image
    # ------------------------------------------------------------------
    if args.output:
        out_path = args.output
    elif isinstance(img_input, str):
        base, ext = os.path.splitext(img_input)
        out_path = base + "_enhanced" + ext
    else:
        out_path = "enhanced_output.png"

    cv2.imwrite(out_path, results["enhanced"])
    print(f"[Info] Enhanced image saved to: {out_path}")

    # ------------------------------------------------------------------
    # Print metrics
    # ------------------------------------------------------------------
    metrics = compute_metrics(results["original"], results["enhanced"])
    print("\n── Metrics ─────────────────────────────────────────────────")
    print(f"  MSE                    : {metrics['mse']:.2f}")
    print(f"  PSNR                   : {metrics['psnr']:.2f} dB")
    o_m = metrics["original_channel_means"]
    e_m = metrics["enhanced_channel_means"]
    print(f"  Original channel means : B={o_m[0]:.1f}  G={o_m[1]:.1f}  R={o_m[2]:.1f}")
    print(f"  Enhanced channel means : B={e_m[0]:.1f}  G={e_m[1]:.1f}  R={e_m[2]:.1f}")
    print(f"  Red channel gain       : {metrics['red_channel_gain']:.2f}×")
    print(f"  Colorfulness (orig.)   : {metrics['original_colorfulness']:.2f}")
    print(f"  Colorfulness (enh.)    : {metrics['enhanced_colorfulness']:.2f}")

    bg = results["background_light"]
    print(f"  Background light (B/G/R): [{bg[0]:.3f}, {bg[1]:.3f}, {bg[2]:.3f}]")
    print("─────────────────────────────────────────────────────────────\n")

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------
    visualize_results(results, save_path=args.save_fig)


if __name__ == "__main__":
    main()
