"""
ReActor V3 - Adaptive Face Enhancement
==========================================

Identity-preserving, fully dynamic face optimization.

All corrections are:
  - Derived from measurable metric error
  - Bounded by face-type classification
  - Scaled by identity similarity
  - Adjusted for luminance variance
  - Operated in LAB for color

No hardcoded sharpening amount.
No fixed restore weight.
No global histogram shift.
Everything adaptive.

Pipeline (per the adaptive enhancement spec):
  Swap
  -> Compute metrics (ref, target, swapped)
  -> Face-type classification
  -> LAB masked luminance histogram match
  -> Adaptive sharpening (HF ratio based)
  -> Controlled texture injection
  -> GPEN restore (identity-aware scaled) [handled by swapper]
  -> Lighting-aware scaling
  -> Confidence-based blending
  -> Final micro-correction if needed
"""

import math
import dataclasses
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict

import cv2
import numpy as np


def _cosine_similarity(a, b):
    """Cosine similarity between two embedding vectors."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ──────────────────────────────────────────────────────────────────────────────
# 0.  ADAPTIVE CONFIG — no hardcoded single value anywhere
# ──────────────────────────────────────────────────────────────────────────────

ADAPTIVE_CONFIG = {
    "identity_weight": 0.6,
    "max_sharpen": 0.50,
    "max_texture_blend": 0.55,
    "low_lum_threshold": 45,
    "restore_min": 0.2,
    "restore_max": 0.8,
    "similarity_scale": 8.0,
    "chroma_align_strength": 0.18,  # raised 0.08→0.18: 0.08 was too weak for scene-lighting variation in SDXL outputs
    "small_face_px": 300,
    "confidence_threshold": 0.4,
}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _safe_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]


def _face_bbox_expanded(face, img_shape, expand_ratio: float = 0.2) -> Tuple[int, int, int, int]:
    """Return face bbox expanded by expand_ratio, clamped to image bounds."""
    if not hasattr(face, 'bbox') or face.bbox is None:
        h, w = img_shape[:2]
        return 0, 0, w, h
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    fw, fh = x2 - x1, y2 - y1
    pad_x = int(fw * expand_ratio)
    pad_y = int(fh * expand_ratio)
    h, w = img_shape[:2]
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    x = float(np.clip(x, -30, 30))
    return 1.0 / (1.0 + math.exp(-x))


def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    if emb1 is None or emb2 is None:
        return 0.0
    emb1 = np.asarray(emb1, dtype=np.float32).flatten()
    emb2 = np.asarray(emb2, dtype=np.float32).flatten()
    norm1 = float(np.linalg.norm(emb1))
    norm2 = float(np.linalg.norm(emb2))
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(emb1 / norm1, emb2 / norm2))


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Face Detail Metrics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceDetailMetrics:
    """Quantitative face detail measurements — used for adaptive corrections."""

    sharpness: float = 0.0           # Laplacian variance (higher = sharper)
    texture_energy: float = 0.0      # mean |Laplacian|  (higher = more detail)
    local_contrast: float = 0.0      # luminance std
    skin_texture: float = 0.0        # multi-orientation Gabor energy
    hf_ratio: float = 0.0            # high-freq energy / total energy
    edge_density: float = 0.0        # fraction of Canny edge pixels
    mean_luminance: float = 0.0      # mean gray value
    luminance_variance: float = 0.0  # variance of L channel (lighting measure)

    def __repr__(self):
        return (
            f"FaceDetail(sharp={self.sharpness:.1f}, tex={self.texture_energy:.1f}, "
            f"contrast={self.local_contrast:.1f}, skin={self.skin_texture:.1f}, "
            f"hf={self.hf_ratio:.3f}, edges={self.edge_density:.3f}, "
            f"lum={self.mean_luminance:.1f}, lum_var={self.luminance_variance:.1f})"
        )

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


class FaceDetailAnalyzer:
    """Extract FaceDetailMetrics from a face crop (BGR uint8)."""

    GABOR_THETAS  = [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]
    GABOR_LAMBDAS = [6.0, 10.0]
    GABOR_SIGMA   = 3.0
    GABOR_GAMMA   = 0.5
    GABOR_KSIZE   = (15, 15)
    
    # Pre-computed Gabor kernels cache (class-level, shared across instances)
    _gabor_kernels_cache = None
    
    @classmethod
    def _get_gabor_kernels(cls):
        """Lazy-initialize and return cached Gabor kernels to avoid recomputing on every analyze() call."""
        if cls._gabor_kernels_cache is None:
            cls._gabor_kernels_cache = []
            for theta in cls.GABOR_THETAS:
                for lambd in cls.GABOR_LAMBDAS:
                    kernel = cv2.getGaborKernel(
                        cls.GABOR_KSIZE, cls.GABOR_SIGMA, theta,
                        lambd, cls.GABOR_GAMMA, psi=0, ktype=cv2.CV_32F
                    )
                    cls._gabor_kernels_cache.append(kernel)
        return cls._gabor_kernels_cache

    def analyze(self, face_crop: np.ndarray) -> FaceDetailMetrics:
        if face_crop is None or face_crop.size == 0:
            return FaceDetailMetrics()

        gray = _to_gray(face_crop).astype(np.float32)
        m = FaceDetailMetrics()

        # Sharpness (Laplacian variance)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        m.sharpness = float(np.var(lap))

        # Texture energy (mean |Laplacian|)
        m.texture_energy = float(np.mean(np.abs(lap)))

        # Local contrast (luminance std)
        m.local_contrast = float(np.std(gray))

        # Mean luminance
        m.mean_luminance = float(np.mean(gray))

        # Luminance variance (for lighting-aware scaling)
        m.luminance_variance = float(np.var(gray))

        # Gabor skin texture energy (L channel) — using pre-cached kernels
        if face_crop.ndim == 3:
            lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
            L = lab[:, :, 0].astype(np.float32)
        else:
            L = gray

        gabor_kernels = self._get_gabor_kernels()
        gabor_responses = []
        for kernel in gabor_kernels:
            resp = cv2.filter2D(L, cv2.CV_32F, kernel)
            gabor_responses.append(float(np.mean(np.abs(resp))))
        m.skin_texture = float(np.mean(gabor_responses))

        # High-frequency energy ratio
        blurred_fine = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
        hf_fine = gray - blurred_fine
        total_energy = float(np.mean(gray ** 2)) + 1e-6
        hf_energy = float(np.mean(hf_fine ** 2))
        m.hf_ratio = hf_energy / total_energy

        # Edge density (Canny)
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        edges = cv2.Canny(gray_u8, 50, 150)
        m.edge_density = float(np.mean(edges > 0))

        return m


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Face-Type Classification  (Section H)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceTypeInfo:
    """Processing bounds per face-type class."""
    face_type: str = "normal"       # "high_quality" | "normal" | "low_res"
    max_sharpen: float = 0.25
    max_restore: float = 0.6
    max_texture: float = 0.3

    def __repr__(self):
        return (
            f"FaceType({self.face_type}, max_sharp={self.max_sharpen:.2f}, "
            f"max_restore={self.max_restore:.2f}, max_tex={self.max_texture:.2f})"
        )


def classify_face_type(metrics: FaceDetailMetrics) -> FaceTypeInfo:
    """
    Classify face as high_quality / normal / low_res using measured metrics.
    Define bounded processing limits per type.
    """
    sharp   = metrics.sharpness
    hf      = metrics.hf_ratio

    if sharp > 3000 and hf > 0.005:
        info = FaceTypeInfo(
            face_type="high_quality",
            max_sharpen=0.25,                                     # raised 0.15→0.25: SDXL sources are sharp but still need clarity after inswapper
            max_restore=0.4,                                      # cap: prevent bone distortion
            max_texture=0.2,
        )
    elif sharp < 1000:
        info = FaceTypeInfo(
            face_type="low_res",
            max_sharpen=ADAPTIVE_CONFIG["max_sharpen"],           # allow more
            max_restore=ADAPTIVE_CONFIG["restore_max"],           # allow more
            max_texture=ADAPTIVE_CONFIG["max_texture_blend"],
        )
    else:
        info = FaceTypeInfo(
            face_type="normal",
            max_sharpen=0.40,
            max_restore=0.7,
            max_texture=0.40,
        )

    print(f"[FaceFixer] Face-type classified: {info}")
    return info


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Skin Mask Generation  (for LAB histogram — Section A)
# ──────────────────────────────────────────────────────────────────────────────

def generate_skin_mask(face_crop: np.ndarray) -> np.ndarray:
    """
    Generate a skin-region float mask [0-1] using HSV skin-colour detection.
    Falls back to an elliptical whole-face mask when HSV coverage is too low.
    """
    if face_crop is None or face_crop.size == 0:
        return np.ones(face_crop.shape[:2], dtype=np.float32)

    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)

    # Primary skin range
    mask1 = cv2.inRange(hsv, np.array([0, 20, 50], np.uint8),
                               np.array([35, 255, 255], np.uint8))
    # Secondary (reddish) skin range
    mask2 = cv2.inRange(hsv, np.array([155, 20, 50], np.uint8),
                               np.array([180, 255, 255], np.uint8))
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 2.0)
    mask_f = mask.astype(np.float32) / 255.0

    # If coverage too low, fall back to elliptical mask
    if float(np.mean(mask_f > 0.3)) < 0.15:
        h, w = face_crop.shape[:2]
        mask_f = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        axes = (max(2, int(w * 0.4)), max(2, int(h * 0.4)))
        cv2.ellipse(mask_f, center, axes, 0, 0, 360, 1.0, -1)
        mask_f = cv2.GaussianBlur(mask_f, (7, 7), 2.0)

    return mask_f


# ──────────────────────────────────────────────────────────────────────────────
# 5A.  Masked LAB Histogram Adaptation   (Section A)
#      Replaces mean colour shift with masked luminance histogram match.
# ──────────────────────────────────────────────────────────────────────────────

def _histogram_match_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match histogram of *source* to *reference* (both 1-D float32 arrays)."""
    if source.size == 0 or reference.size == 0:
        return source
    src_vals = source.flatten()
    ref_vals = reference.flatten()

    src_sorted = np.sort(src_vals)
    ref_sorted = np.sort(ref_vals)

    src_cdf = np.linspace(0, 1, len(src_sorted))
    ref_cdf = np.linspace(0, 1, len(ref_sorted))

    interp_values = np.interp(src_cdf, ref_cdf, ref_sorted)
    src_indices = np.argsort(np.argsort(src_vals))
    return interp_values[src_indices].reshape(source.shape)


def apply_lab_histogram_adaptation(
    swapped_crop: np.ndarray,
    target_crop: np.ndarray,
    skin_mask: np.ndarray,
) -> np.ndarray:
    """
    Transfer **lighting** (not skin tone) from target face to swapped face.

    ● L channel: full histogram match (luminance transfer)
    ● A/B channels: light mean alignment only (chroma nudge, NOT full histogram)
    ● Operates only inside skin_mask
    ● All arithmetic in float32 before clipping
    """
    if swapped_crop is None or target_crop is None:
        return swapped_crop

    h, w = swapped_crop.shape[:2]
    if target_crop.shape[:2] != (h, w):
        target_crop = cv2.resize(target_crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if skin_mask.shape[:2] != (h, w):
        skin_mask = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert to LAB float32
    lab_swap = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_tgt  = cv2.cvtColor(target_crop, cv2.COLOR_BGR2LAB).astype(np.float32)

    mask_bool = skin_mask > 0.3
    if int(np.sum(mask_bool)) < 50:
        return swapped_crop

    # ── L channel: conservative mean brightness alignment only ──────────
    # Full histogram match was identity-destructive: it completely remapped
    # shadow patterns (a strong identity cue), making the face read as a
    # different person. Instead shift only the mean brightness, capped at
    # ±12 LAB units so dramatic lighting differences can't flip identity.
    L_swap_masked = lab_swap[:, :, 0][mask_bool]
    L_tgt_masked  = lab_tgt[:, :, 0][mask_bool]
    L_delta = float(np.mean(L_tgt_masked)) - float(np.mean(L_swap_masked))
    L_delta = float(np.clip(L_delta, -18.0, 18.0))  # widened cap: ±12 was too tight for dramatic scene lighting

    lab_result = lab_swap.copy()
    lab_result[:, :, 0][mask_bool] = np.clip(
        lab_swap[:, :, 0][mask_bool] + L_delta, 0.0, 255.0
    )

    # ── A/B channels: minimal nudge (0.08) ─────────────────────────────
    # 0.45 was shifting skin colour 45% toward the target person's tone,
    # producing an in-between face that looked like neither source nor target.
    chroma_str = ADAPTIVE_CONFIG["chroma_align_strength"]  # 0.08
    for ch in [1, 2]:
        swap_mean = float(np.mean(lab_swap[:, :, ch][mask_bool]))
        tgt_mean  = float(np.mean(lab_tgt[:, :, ch][mask_bool]))
        lab_result[:, :, ch][mask_bool] += chroma_str * (tgt_mean - swap_mean)

    # Clip in float32 then convert back
    lab_result = np.clip(lab_result, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

    # Smooth blend at mask edges
    mask_3 = skin_mask[:, :, None]
    blended = result.astype(np.float32) * mask_3 + swapped_crop.astype(np.float32) * (1.0 - mask_3)
    result = np.clip(blended, 0, 255).astype(np.uint8)

    print("[FaceFixer] LAB histogram adaptation applied (L-histogram + A/B mean align)")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5B.  Identity-Aware Restore Weight Scaling   (Section B — Critical)
# ──────────────────────────────────────────────────────────────────────────────

def compute_identity_restore_weight(
    source_embedding: np.ndarray,
    swapped_embedding: np.ndarray,
    restored_embedding: np.ndarray,
    base_restore_weight: float = 0.8,
) -> float:
    """
    Adaptive GPEN restore weight based on identity-similarity feedback.

    If GPEN damages identity (similarity drops) → reduce restore weight.
    If identity collapses → drop restore to near zero.

    Returns:
        Clamped restore weight in [restore_min, restore_max].
    """
    if source_embedding is None or swapped_embedding is None or restored_embedding is None:
        return base_restore_weight

    sim_pre  = _cosine_similarity(source_embedding, swapped_embedding)
    sim_post = _cosine_similarity(source_embedding, restored_embedding)
    delta_sim = sim_post - sim_pre

    identity_penalty = _sigmoid(-delta_sim * 10)
    restore_weight = base_restore_weight * (1.0 - identity_penalty)

    # Step 5: Hard identity-drift check — halve weight if similarity drops >0.02
    if delta_sim < -0.02:
        restore_weight *= 0.5
        print(f"[FaceFixer] Identity drift detected (delta={delta_sim:.3f} < -0.02) — halving restore weight")

    restore_weight = float(np.clip(
        restore_weight,
        ADAPTIVE_CONFIG["restore_min"],
        ADAPTIVE_CONFIG["restore_max"],
    ))

    print(
        f"[FaceFixer] Identity-aware restore: sim_pre={sim_pre:.3f}, "
        f"sim_post={sim_post:.3f}, delta={delta_sim:.3f}, "
        f"penalty={identity_penalty:.3f}, weight={restore_weight:.3f}"
    )
    return restore_weight


# ──────────────────────────────────────────────────────────────────────────────
# 5C.  Adaptive Sharpening   (Section C — replaces static unsharp mask)
# ──────────────────────────────────────────────────────────────────────────────

def compute_hf_energy(face_crop: np.ndarray, sigma: float = 2.8) -> float:
    """Perceptual mid-frequency energy (bandpass at ~2-5px scale).

    sigma=2.8 targets the clarity band that makes faces look sharp vs.
    AI-blurry. The old sigma=1.5 measured sub-pixel noise instead, causing
    hf_ratio≈1.0 for blurry inswapper outputs and gain≈0 (no sharpening).
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    gray = _to_gray(face_crop).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    hf = gray - blurred
    return float(np.mean(hf ** 2))


def apply_adaptive_sharpening(
    swapped_crop: np.ndarray,
    ref_crop: np.ndarray,
    face_type: FaceTypeInfo,
    lum_variance: float,
) -> np.ndarray:
    """
    Proportional sharpening derived from HF-energy ratio.

    ● Never sharpens already-sharp faces (hf_ratio >= 1).
    ● Gain is proportional to (1 - hf_ratio), capped by face_type.
    ● Reduced in low-luminance / low-contrast regions  (Section E).
    ● Edge-aware: suppresses sharpening near strong edges to avoid halos.
    """
    if swapped_crop is None or ref_crop is None:
        return swapped_crop

    h, w = swapped_crop.shape[:2]
    ref_resized = cv2.resize(ref_crop, (w, h), interpolation=cv2.INTER_LANCZOS4)

    hf_ref  = compute_hf_energy(ref_resized)
    hf_swap = compute_hf_energy(swapped_crop)

    if hf_ref < 1e-6:
        return swapped_crop

    hf_ratio = hf_swap / hf_ref

    if hf_ratio >= 1.0:
        print(f"[FaceFixer] Adaptive sharpen: hf_ratio={hf_ratio:.3f} >= 1.0 — skipping")
        return swapped_crop

    # Proportional gain capped by face-type limit
    # Multiplier 1.8: lower than 2.2 to prevent over-sharpening on blurry
    # inswapper output while still giving a meaningful gain (0.9×) at 50 % deficit.
    gain = float(np.clip((1.0 - hf_ratio) * 1.8, 0.0, face_type.max_sharpen))

    # Lighting-aware scaling  (Section E)
    if lum_variance < ADAPTIVE_CONFIG["low_lum_threshold"]:
        gain *= 0.5

    if gain < 0.01:
        return swapped_crop

    # Edge-aware unsharp mask — sigma=2.0 targets the perceptual clarity band
    swap_f  = swapped_crop.astype(np.float32)
    blurred = cv2.GaussianBlur(swap_f, (0, 0), sigmaX=2.0)
    detail  = swap_f - blurred

    gray  = _to_gray(swapped_crop)
    edges = cv2.Canny(gray, 50, 150)
    edge_mask = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (5, 5), 1.5)
    edge_suppress = 1.0 - edge_mask * 0.5
    if detail.ndim == 3:
        edge_suppress = edge_suppress[:, :, None]

    result = swap_f + detail * gain * edge_suppress
    result = np.clip(result, 0, 255).astype(np.uint8)

    print(f"[FaceFixer] Adaptive sharpen: hf_ratio={hf_ratio:.3f}, gain={gain:.3f}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5D.  Controlled Texture Injection   (Section D)
# ──────────────────────────────────────────────────────────────────────────────

def apply_texture_injection(
    swapped_crop: np.ndarray,
    ref_crop: np.ndarray,
    skin_mask: np.ndarray,
    face_type: FaceTypeInfo,
    lum_variance: float,
) -> np.ndarray:
    """
    Extract zero-mean HF layer from reference, warp to swapped alignment,
    and blend into swapped face — skin-masked only.

    Blend strength proportional to HF deficit, bounded by face_type,
    reduced in low-luminance regions  (Section E).
    """
    if swapped_crop is None or ref_crop is None:
        return swapped_crop

    h, w = swapped_crop.shape[:2]
    ref_resized = cv2.resize(ref_crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if skin_mask.shape[:2] != (h, w):
        skin_mask = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    hf_ref  = compute_hf_energy(ref_resized)
    hf_swap = compute_hf_energy(swapped_crop)

    if hf_ref < 1e-6:
        return swapped_crop

    hf_ratio = hf_swap / hf_ref
    if hf_ratio >= 1.0:
        return swapped_crop

    blend_strength = float(np.clip((1.0 - hf_ratio) * 0.8, 0.0, face_type.max_texture))

    # Lighting-aware scaling (Section E)
    if lum_variance < ADAPTIVE_CONFIG["low_lum_threshold"]:
        blend_strength *= 0.5

    if blend_strength < 0.01:
        return swapped_crop

    # Extract zero-mean HF from reference
    ref_f  = ref_resized.astype(np.float32)
    ref_lf = cv2.GaussianBlur(ref_f, (0, 0), sigmaX=2.0)
    hf_ref_layer = ref_f - ref_lf

    # Zero-mean per channel
    if hf_ref_layer.ndim == 3:
        for ch in range(hf_ref_layer.shape[2]):
            hf_ref_layer[:, :, ch] -= np.mean(hf_ref_layer[:, :, ch])
    else:
        hf_ref_layer -= np.mean(hf_ref_layer)

    # Blend into swapped (skin mask only)
    swap_f = swapped_crop.astype(np.float32)
    mask_3 = skin_mask[:, :, None] if swap_f.ndim == 3 else skin_mask
    swap_f += blend_strength * hf_ref_layer * mask_3

    result = np.clip(swap_f, 0, 255).astype(np.uint8)
    print(f"[FaceFixer] Texture injection: hf_ratio={hf_ratio:.3f}, blend={blend_strength:.3f}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5F.  Resolution-Aware Restore Limiting   (Section F)
# ──────────────────────────────────────────────────────────────────────────────

def compute_resolution_restore_limit(face_bbox) -> float:
    """
    If face bbox < 300 px → cap restore_weight at 0.4.
    Prevents hallucinated detail on small faces.
    """
    if face_bbox is None:
        return 1.0
    try:
        bbox = list(face_bbox)
        if len(bbox) >= 4:
            face_size = max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))
        else:
            return 1.0
    except Exception:
        return 1.0

    small_px = ADAPTIVE_CONFIG["small_face_px"]
    if face_size < small_px:
        limit = 0.4
        print(f"[FaceFixer] Resolution limit: face={face_size:.0f}px < {small_px}px — capping restore at {limit}")
        return limit
    return 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 5G.  Confidence-Based Global Blending   (Section G)
# ──────────────────────────────────────────────────────────────────────────────

def compute_confidence_blend(
    source_embedding: np.ndarray,
    result_face_embedding: np.ndarray,
) -> float:
    """
    Sigmoid-scaled cosine similarity → confidence.
    Low confidence → blend more of original target back.
    """
    if source_embedding is None or result_face_embedding is None:
        return 1.0

    similarity = _cosine_similarity(source_embedding, result_face_embedding)
    threshold  = ADAPTIVE_CONFIG["confidence_threshold"]
    scale      = ADAPTIVE_CONFIG["similarity_scale"]
    confidence = _sigmoid((similarity - threshold) * scale)

    print(f"[FaceFixer] Confidence blend: similarity={similarity:.3f}, confidence={confidence:.3f}")
    return float(confidence)


def apply_confidence_blending(
    processed_face: np.ndarray,
    original_target_face: np.ndarray,
    confidence: float,
) -> np.ndarray:
    """
    final = confidence * processed + (1 - confidence) * original_target
    Prevents heavy processing on weak matches.
    """
    if confidence >= 0.99:
        return processed_face
    if original_target_face is None:
        return processed_face

    h, w = processed_face.shape[:2]
    if original_target_face.shape[:2] != (h, w):
        original_target_face = cv2.resize(original_target_face, (w, h), interpolation=cv2.INTER_LANCZOS4)

    result = (
        confidence * processed_face.astype(np.float32)
        + (1.0 - confidence) * original_target_face.astype(np.float32)
    )
    return np.clip(result, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Main Face Detail Fixer — full adaptive pipeline
# ──────────────────────────────────────────────────────────────────────────────

class FaceDetailFixer:
    """
    Adaptive face quality fixer.

    Replaces all static constants with measurement-driven adaptive corrections
    that are proportional to error, bounded by face-type, scaled by identity
    similarity, adjusted for luminance variance, and operated in LAB for color.
    """

    def __init__(self):
        self.analyzer = FaceDetailAnalyzer()

    # ─────────────────────────────────────────────────────────────────────
    def fix(
        self,
        reference_img: np.ndarray,
        output_img: np.ndarray,
        face_analyser=None,
        ref_face=None,
        out_face=None,
        fix_all_faces: bool = False,
        target_img: np.ndarray = None,
        source_face=None,
        target_face=None,
    ) -> np.ndarray:
        """
        Full adaptive face enhancement pipeline.

        Args:
            reference_img : Source/reference image — identity anchor (BGR uint8).
            output_img    : Swapped/restored output — working state (BGR uint8).
            face_analyser : InsightFace FaceAnalysis instance.
            ref_face      : Pre-detected reference face object (optional).
            out_face      : Pre-detected output face object (optional).
            fix_all_faces : Fix every detected output face.
            target_img    : Original target image — lighting anchor (BGR uint8).
            source_face   : Source face object with .embedding (for confidence).

        Returns:
            Fixed output image (BGR uint8).
        """
        if reference_img is None or output_img is None:
            return output_img

        print("")
        print("[FaceFixer] ══════════════════════════════════════════")
        print("[FaceFixer]   ADAPTIVE FACE ENHANCEMENT START")
        print("[FaceFixer] ══════════════════════════════════════════")

        # ── Detect faces if not provided ──────────────────────────────────
        if ref_face is None and face_analyser is not None:
            ref_faces = face_analyser.get(reference_img)
            if ref_faces:
                # Prefer the face closest by embedding to source_face if available
                if source_face is not None and hasattr(source_face, 'embedding') and source_face.embedding is not None:
                    best_sim = -1.0
                    for rf in ref_faces:
                        if hasattr(rf, 'embedding') and rf.embedding is not None:
                            sim = _cosine_similarity(source_face.embedding, rf.embedding)
                            if sim > best_sim:
                                best_sim = sim
                                ref_face = rf
                    if ref_face is None:
                        ref_face = ref_faces[0]
                    print(f"[FaceFixer] Reference face matched by embedding (sim={best_sim:.3f}): bbox={[int(v) for v in ref_face.bbox]}")
                else:
                    # Fallback: largest face
                    ref_face = max(
                        ref_faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                        if hasattr(f, 'bbox') else 0
                    )
                    print(f"[FaceFixer] Reference face detected (largest): bbox={[int(v) for v in ref_face.bbox]}")
            else:
                print("[FaceFixer] WARNING: No face in reference image — skipping")
                return output_img

        if out_face is None and face_analyser is not None:
            out_faces = face_analyser.get(output_img)
            if not out_faces:
                print("[FaceFixer] WARNING: No face in output image — skipping")
                return output_img
        else:
            out_faces = [out_face] if out_face is not None else []

        if not out_faces or ref_face is None:
            print("[FaceFixer] No faces to process — skipping")
            return output_img

        # ── Reference crop & metrics ──────────────────────────────────────
        ref_bbox = _face_bbox_expanded(ref_face, reference_img.shape, expand_ratio=0.15)
        ref_crop = _safe_crop(reference_img, *ref_bbox)
        ref_metrics = self.analyzer.analyze(ref_crop)
        print(f"[FaceFixer] Reference metrics: {ref_metrics}")

        # ── Source embedding (for confidence blending) ────────────────────
        source_emb = None
        if source_face is not None and hasattr(source_face, 'embedding'):
            source_emb = source_face.embedding
        elif ref_face is not None and hasattr(ref_face, 'embedding'):
            source_emb = ref_face.embedding

        # ── Target faces (lighting anchor) ───────────────────────────────
        # Use cached target_face if provided to avoid redundant detection
        _tgt_faces_cache = []
        if target_face is not None and hasattr(target_face, 'bbox'):
            _tgt_faces_cache = [target_face]
            print("[FaceFixer] Using pre-detected target face (cached) — no re-detection")
        elif target_img is not None and face_analyser is not None:
            _tgt_faces_cache = face_analyser.get(target_img) or []
            print(f"[FaceFixer] Detected {len(_tgt_faces_cache)} target face(s) for lighting anchor")

        # ── Fix each output face ──────────────────────────────────────────
        result = output_img.copy()
        faces_to_fix = out_faces if fix_all_faces else out_faces[:1]

        for i, oface in enumerate(faces_to_fix):
            if oface is None or not hasattr(oface, 'bbox'):
                continue

            out_bbox = _face_bbox_expanded(oface, output_img.shape, expand_ratio=0.15)
            out_crop = _safe_crop(output_img, *out_bbox)

            # Save original crop for confidence blending later
            original_out_crop = out_crop.copy()

            out_metrics = self.analyzer.analyze(out_crop)
            print(f"[FaceFixer] Output face {i} metrics: {out_metrics}")

            # ── Per-face target crop (lighting anchor) ───────────────────
            # Match this output face to the closest target face by bbox center
            target_crop = None
            if _tgt_faces_cache:
                o_cx = (oface.bbox[0] + oface.bbox[2]) / 2
                o_cy = (oface.bbox[1] + oface.bbox[3]) / 2
                best_dist = float('inf')
                best_tgt = None
                for tf in _tgt_faces_cache:
                    if not hasattr(tf, 'bbox'):
                        continue
                    t_cx = (tf.bbox[0] + tf.bbox[2]) / 2
                    t_cy = (tf.bbox[1] + tf.bbox[3]) / 2
                    dist = (o_cx - t_cx) ** 2 + (o_cy - t_cy) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_tgt = tf
                if best_tgt is not None:
                    tgt_bbox = _face_bbox_expanded(best_tgt, target_img.shape, expand_ratio=0.15)
                    target_crop = _safe_crop(target_img, *tgt_bbox)
                    print(f"[FaceFixer] Target lighting anchor for face {i}: bbox={[int(v) for v in best_tgt.bbox]}, dist={best_dist:.0f}")

            # ── 1. Face-type classification  (H) ─────────────────────────
            face_type = classify_face_type(ref_metrics)

            # ── 2. Skin mask ─────────────────────────────────────────────
            skin_mask = generate_skin_mask(out_crop)

            # ── 3. LAB masked luminance histogram match  (A) ─────────────
            if target_crop is not None:
                out_crop = apply_lab_histogram_adaptation(out_crop, target_crop, skin_mask)
            else:
                print("[FaceFixer] No target image — skipping LAB histogram adaptation")

            # ── 4. Adaptive sharpening  (C + E) ─────────────────────────
            out_crop = apply_adaptive_sharpening(
                out_crop, ref_crop, face_type, out_metrics.luminance_variance,
            )

            # ── 5. Texture injection disabled ─────────────────────────
            # Injecting source-face HF texture onto the swapped face directly
            # causes ghosted/double-face artifacts (reference skin pattern bleeds
            # through). GPEN's HF delta already handles detail recovery.

            # ── 6. Confidence-based blending  (G) ────────────────────────
            if source_emb is not None and face_analyser is not None:
                # Get embedding of the processed face to measure similarity
                try:
                    proc_faces = face_analyser.get(out_crop)
                    if proc_faces:
                        proc_emb = proc_faces[0].embedding
                        confidence = compute_confidence_blend(source_emb, proc_emb)
                        out_crop = apply_confidence_blending(
                            out_crop, original_out_crop, confidence,
                        )
                except Exception as e:
                    print(f"[FaceFixer] Confidence blending skipped: {e}")

            # ── 7. Paste back with feathered mask ────────────────────────
            result = self._paste_back(result, out_crop, out_bbox, oface)
            print(f"[FaceFixer] Face {i}: Adaptive enhancement applied ✓")

        print("[FaceFixer] ══════════════════════════════════════════")
        print("")
        return result

    # ─────────────────────────────────────────────────────────────────────
    def _paste_back(
        self,
        full_img: np.ndarray,
        fixed_crop: np.ndarray,
        bbox: Tuple[int, int, int, int],
        face,
    ) -> np.ndarray:
        """Paste with an elliptical feathered + landmark-refined mask."""
        x1, y1, x2, y2 = bbox
        h, w = full_img.shape[:2]

        expected_w = x2 - x1
        expected_h = y2 - y1
        if fixed_crop.shape[1] != expected_w or fixed_crop.shape[0] != expected_h:
            fixed_crop = cv2.resize(
                fixed_crop, (expected_w, expected_h), interpolation=cv2.INTER_LANCZOS4
            )

        # Elliptical base mask
        mask = np.zeros((expected_h, expected_w), dtype=np.float32)
        center = (expected_w // 2, expected_h // 2)
        axes = (max(2, int(expected_w * 0.48)), max(2, int(expected_h * 0.48)))  # widened 0.45→0.48 to match GPEN composite boundary
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        # Feather
        feather = max(7, int(min(expected_w, expected_h) * 0.18))
        if feather % 2 == 0:
            feather += 1
        mask = cv2.GaussianBlur(mask, (feather, feather), feather * 0.35)
        mask = np.clip(mask, 0.0, 1.0)

        # Landmark refinement
        if hasattr(face, 'kps') and face.kps is not None:
            kps = np.asarray(face.kps, dtype=np.float32)
            if kps.ndim == 2 and kps.shape[0] >= 5:
                kps_local = kps.copy()
                kps_local[:, 0] -= x1
                kps_local[:, 1] -= y1
                kps_local[:, 0] = np.clip(kps_local[:, 0], 0, expected_w - 1)
                kps_local[:, 1] = np.clip(kps_local[:, 1], 0, expected_h - 1)

                lm_mask = np.zeros_like(mask)
                hull = cv2.convexHull(kps_local.astype(np.int32))
                cv2.fillConvexPoly(lm_mask, hull, 1.0)
                dilate_k = max(3, int(min(expected_w, expected_h) * 0.12))
                kernel_d = np.ones((dilate_k, dilate_k), dtype=np.uint8)
                lm_mask = cv2.dilate(lm_mask, kernel_d, iterations=1)
                lm_mask = cv2.GaussianBlur(lm_mask, (feather, feather), feather * 0.25)
                mask = np.maximum(mask, lm_mask * 0.85)
                mask = np.clip(mask, 0.0, 1.0)

        # Blend into full image
        result = full_img.copy()
        region = result[y1:y2, x1:x2]
        mask_3 = mask[:, :, None]
        blended = fixed_crop.astype(np.float32) * mask_3 + region.astype(np.float32) * (1.0 - mask_3)
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        return result


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Convenience one-call function
# ──────────────────────────────────────────────────────────────────────────────

_fixer_instance: Optional[FaceDetailFixer] = None


def get_face_fixer() -> FaceDetailFixer:
    global _fixer_instance
    if _fixer_instance is None:
        _fixer_instance = FaceDetailFixer()
    return _fixer_instance


def auto_fix_face(
    reference_img: np.ndarray,
    output_img: np.ndarray,
    face_analyser=None,
    ref_face=None,
    out_face=None,
    target_img: np.ndarray = None,
    source_face=None,
    target_face=None,
) -> np.ndarray:
    """
    One-call adaptive face enhancement.

    Args:
        reference_img : Source photo — identity anchor (BGR uint8).
        output_img    : Swapped/restored output (BGR uint8).
        face_analyser : InsightFace FaceAnalysis instance.
        ref_face      : Pre-detected reference face (optional).
        out_face      : Pre-detected output face (optional).
        target_img    : Original target image — lighting anchor (BGR uint8).
        source_face   : Source face with .embedding (for confidence blend).
        target_face   : The specific swapped face to enhance — skips full-image
                        re-detection and processes only this face. Pass the face
                        object returned by InsightFace after swap.

    Returns:
        Adaptively enhanced output image (BGR uint8).
    """
    fixer = get_face_fixer()
    return fixer.fix(
        reference_img=reference_img,
        output_img=output_img,
        face_analyser=face_analyser,
        ref_face=ref_face,
        out_face=out_face if out_face is not None else target_face,
        target_img=target_img,
        source_face=source_face,
        target_face=target_face,
    )
