"""
ReActor V3 - Adaptive Pipeline
================================
Fully adaptive face-swap pipeline with feedback loop.

Stages:
  1. Analyze source face quality (blur · noise · exposure · pose · skin detail)
  2. Auto-select pipeline params from quality scores
  3. Run first-pass swap + restoration
  4. Detect output artifacts  (plastic skin · grain · edge seam)
  5. Retry with corrected params  (max MAX_RETRIES passes)
  6. Fallback / flag for manual review when confidence is too low

Usage (standalone):
    from reactor_v3_adaptive import AdaptiveReActorPipeline
    pipeline = AdaptiveReActorPipeline(engine)
    result_img, report = pipeline.run(source_img, target_img, ...)
"""

import math
import dataclasses
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Shared micro-utilities
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


def _face_region(img: np.ndarray, face) -> np.ndarray:
    """Return the face crop from an InsightFace face object."""
    if face is None or not hasattr(face, "bbox"):
        return img
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    return _safe_crop(img, x1, y1, x2, y2)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Face Quality Analyzer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceQualityScores:
    """All scores are normalised 0 → 1  (1 = best / cleanest)."""

    blur_score: float       = 1.0   # 1 → sharp,  0 → very blurry
    noise_score: float      = 1.0   # 1 → clean,  0 → very noisy
    exposure_score: float   = 1.0   # 1 → well-exposed, 0 → over/under
    pose_score: float       = 1.0   # 1 → frontal, 0 → extreme angle
    skin_detail_score: float = 1.0  # 1 → rich texture, 0 → flat/smooth
    detect_confidence: float = 1.0  # raw InsightFace det_score (0–1)

    # Derived composite (filled in by analyzer)
    overall: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)

    def __repr__(self):
        return (
            f"FaceQuality("
            f"blur={self.blur_score:.2f}, noise={self.noise_score:.2f}, "
            f"exposure={self.exposure_score:.2f}, pose={self.pose_score:.2f}, "
            f"skin={self.skin_detail_score:.2f}, det_conf={self.detect_confidence:.2f}, "
            f"overall={self.overall:.2f})"
        )


class FaceQualityAnalyzer:
    """
    Computes per-dimension quality scores from a face crop.

    - blur      : Laplacian variance  (tenengrad variant)
    - noise     : High-pass RMS after bilateral smoothing
    - exposure  : Histogram peak check (blown highlights + crushed shadows)
    - pose      : Yaw / pitch from InsightFace pose attribute or landmark geometry
    - skin_det  : Gabor-filter texture energy in L*a*b* L-channel
    """

    # ---------- tunable thresholds ----------
    BLUR_SHARP_VAR   = 400.0   # Laplacian variance → 100% sharp score
    BLUR_BLUR_VAR    = 20.0    # Laplacian variance → 0% sharp score
    NOISE_CLEAN_RMS  = 3.0     # pixel RMS → 0% noise
    NOISE_NOISY_RMS  = 25.0    # pixel RMS → 100% noise
    GABOR_RICH_MEAN  = 18.0    # gabor mean  → 100% skin detail
    GABOR_FLAT_MEAN  = 3.0     # gabor mean  → 0%   skin detail
    OVEREXP_FRAC_MAX = 0.10    # fraction of pixels at 255 that = fully overexposed
    UNDEREXP_FRAC_MAX= 0.10    # fraction of pixels at 0   that = fully underexposed
    POSE_MAX_YAW_DEG = 45.0    # yaw / pitch beyond which score → 0

    def analyze(self, img: np.ndarray, face=None) -> FaceQualityScores:
        """
        Args:
            img   : Full BGR image (numpy uint8).
            face  : InsightFace face object (optional). Used for pose + crop.

        Returns:
            FaceQualityScores
        """
        crop = _face_region(img, face) if face is not None else img
        if crop.size == 0:
            crop = img

        gray = _to_gray(crop)

        blur      = self._score_blur(gray)
        noise     = self._score_noise(gray)
        exposure  = self._score_exposure(gray)
        pose      = self._score_pose(face)
        skin_det  = self._score_skin_detail(crop)
        det_conf  = float(getattr(face, "det_score", 1.0)) if face else 1.0

        overall = (
            blur      * 0.25 +
            noise     * 0.15 +
            exposure  * 0.15 +
            pose      * 0.20 +
            skin_det  * 0.15 +
            det_conf  * 0.10
        )
        overall = float(np.clip(overall, 0.0, 1.0))

        return FaceQualityScores(
            blur_score        = blur,
            noise_score       = noise,
            exposure_score    = exposure,
            pose_score        = pose,
            skin_detail_score = skin_det,
            detect_confidence = det_conf,
            overall           = overall,
        )

    # ------------------------------------------------------------------
    def _score_blur(self, gray: np.ndarray) -> float:
        var = float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())
        s   = (var - self.BLUR_BLUR_VAR) / max(1e-6, self.BLUR_SHARP_VAR - self.BLUR_BLUR_VAR)
        return float(np.clip(s, 0.0, 1.0))

    def _score_noise(self, gray: np.ndarray) -> float:
        """Noise = high-frequency residual after bilateral denoise."""
        smoothed = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
        residual = gray.astype(np.float32) - smoothed.astype(np.float32)
        rms = float(np.sqrt(np.mean(residual ** 2)))
        s   = 1.0 - (rms - self.NOISE_CLEAN_RMS) / max(1e-6, self.NOISE_NOISY_RMS - self.NOISE_CLEAN_RMS)
        return float(np.clip(s, 0.0, 1.0))

    def _score_exposure(self, gray: np.ndarray) -> float:
        total = gray.size
        over  = float(np.sum(gray >= 250)) / total
        under = float(np.sum(gray <=   5)) / total
        over_pen  = (over  / self.OVEREXP_FRAC_MAX )
        under_pen = (under / self.UNDEREXP_FRAC_MAX)
        penalty   = np.clip(over_pen + under_pen, 0.0, 1.0)
        return float(1.0 - penalty)

    def _score_pose(self, face) -> float:
        """Estimate pose from InsightFace pose attr (pitch/yaw/roll in radians or degrees)."""
        if face is None:
            return 1.0
        pose_attr = getattr(face, "pose", None)
        if pose_attr is not None:
            try:
                angles = np.asarray(pose_attr, dtype=np.float32).flatten()
                # InsightFace returns [pitch, yaw, roll] in degrees
                yaw, pitch = float(angles[1]), float(angles[0])
                if abs(yaw) > 5 or abs(pitch) > 5:
                    # looks like degrees already
                    pass
                else:
                    # probably radians
                    yaw   = math.degrees(yaw)
                    pitch = math.degrees(pitch)
                max_angle = max(abs(yaw), abs(pitch))
                s = 1.0 - max_angle / self.POSE_MAX_YAW_DEG
                return float(np.clip(s, 0.0, 1.0))
            except Exception:
                pass
        # Fallback: use face bbox aspect ratio as proxy for off-angle
        if hasattr(face, "bbox"):
            x1, y1, x2, y2 = [float(v) for v in face.bbox]
            w, h = x2 - x1, y2 - y1
            if h > 0:
                ratio = w / h
                # Symmetric faces ≈ 0.7-0.8 aspect ratio
                deviation = abs(ratio - 0.75) / 0.35
                return float(np.clip(1.0 - deviation, 0.0, 1.0))
        return 0.8   # neutral fallback

    def _score_skin_detail(self, crop: np.ndarray) -> float:
        """Gabor texture energy in the luminance (L*) channel."""
        if crop.ndim == 2:
            L = crop.astype(np.float32)
        else:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            L   = lab[:, :, 0].astype(np.float32)

        responses = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel(
                ksize=(15, 15), sigma=3.0, theta=theta,
                lambd=8.0, gamma=0.5, psi=0, ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(L, cv2.CV_32F, kernel)
            responses.append(float(np.mean(np.abs(filtered))))

        gabor_mean = float(np.mean(responses))
        s = (gabor_mean - self.GABOR_FLAT_MEAN) / max(1e-6, self.GABOR_RICH_MEAN - self.GABOR_FLAT_MEAN)
        return float(np.clip(s, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Adaptive Parameters & Selector
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveParams:
    """
    All tuneable knobs for one pipeline pass.
    These are passed back into ReActorV3.process() or applied post-process.
    """

    # Face swap strength  (0 = keep original face, 1 = full swap blend)
    swap_strength: float     = 1.0

    # GPEN restoration strength  (0 = off, 1 = full restoration)
    restore_strength: float  = 0.85

    # Which GPEN model to use  ('None' / 'GPEN-BFR-512.onnx' / 'GPEN-BFR-1024.onnx')
    restore_model: str       = "auto"   # "auto" → picked from quality

    # Enable luminance + colour match between source and target scenes
    color_match: bool        = False

    # How aggressive colour correction is  (0–1)
    color_match_strength: float = 0.5

    # Post-restoration sharpening  (0 = none, 1 = heavy)
    sharpen_strength: float  = 0.3

    # Denoise target before swap  (0 = none, 1 = heavy)
    denoise_strength: float  = 0.0

    # Texture-preservation pass: mix a little of the original back after restoration
    texture_preserve_blend: float = 0.0   # 0 = off, >0 = fraction of original to re-add

    # Internal tag for logging
    reason: str              = ""


class AdaptiveParamSelector:
    """
    Maps FaceQualityScores → AdaptiveParams.

    Decision tree:
      ─ Low detect confidence                    → flag / minimal processing
      ─ Very blurry / low-qual source            → lower swap + restore, no upscale
      ─ High-contrast / hard lighting            → enable colour match, lower CFG-style denoise
      ─ Off-angle pose                           → lower restore strength (geometry distortion risk)
      ─ GPEN model selection                     → 1024 for high quality, 512 for low quality / speed
    """

    QUALITY_HIGH   = 0.70
    QUALITY_MEDIUM = 0.45
    QUALITY_LOW    = 0.25

    BLUR_HIGH   = 0.70
    BLUR_LOW    = 0.35

    NOISE_HIGH  = 0.70
    NOISE_LOW   = 0.35

    CONTRAST_SCORE_HIGH = 0.70   # exposure_score below this → enable colour match

    def select(self, scores: FaceQualityScores,
               available_models: List[str]) -> Tuple[AdaptiveParams, float]:
        """
        Returns:
            (AdaptiveParams, confidence)
            confidence: 0–1.  <0.35 → pipeline will flag for manual review.
        """
        reasons: List[str] = []
        p = AdaptiveParams()

        confidence = scores.overall

        # ── detect confidence gate ──────────────────────────────────────────
        if scores.detect_confidence < 0.50:
            p.swap_strength       = 0.60
            p.restore_strength    = 0.40
            p.restore_model       = self._pick_model(available_models, prefer_1024=False)
            p.texture_preserve_blend = 0.20
            reasons.append("low_detect_confidence")
            confidence *= 0.85

        # ── blur analysis ───────────────────────────────────────────────────
        if scores.blur_score < self.BLUR_LOW:
            p.swap_strength    = min(p.swap_strength,    0.72)
            p.restore_strength = min(p.restore_strength, 0.50)
            # Blurry source → use 512 (don't hallucinate sharpness)
            p.restore_model    = self._pick_model(available_models, prefer_1024=False)
            p.sharpen_strength = 0.0
            reasons.append("source_blurry")
            confidence *= 0.80
        elif scores.blur_score < self.BLUR_HIGH:
            p.swap_strength    = min(p.swap_strength,    0.88)
            p.restore_strength = min(p.restore_strength, 0.75)
            reasons.append("source_slightly_blurry")

        # ── noise analysis ──────────────────────────────────────────────────
        if scores.noise_score < self.NOISE_LOW:
            p.denoise_strength = 0.50
            p.sharpen_strength = 0.0        # don't amplify noise with sharpening
            p.restore_strength = min(p.restore_strength, 0.65)
            reasons.append("source_noisy")
        elif scores.noise_score < self.NOISE_HIGH:
            p.denoise_strength = 0.20
            p.sharpen_strength = min(p.sharpen_strength, 0.25)
            reasons.append("source_mild_noise")

        # ── exposure / contrast analysis ────────────────────────────────────
        if scores.exposure_score < self.CONTRAST_SCORE_HIGH:
            p.color_match          = True
            p.color_match_strength = 0.65
            p.restore_strength     = min(p.restore_strength, 0.70)
            reasons.append("high_contrast_scene:colour_match_on")

        # ── pose angle ──────────────────────────────────────────────────────
        if scores.pose_score < 0.45:
            p.restore_strength       = min(p.restore_strength, 0.55)
            p.texture_preserve_blend = max(p.texture_preserve_blend, 0.15)
            reasons.append("extreme_pose:reduce_restore")
        elif scores.pose_score < 0.65:
            p.restore_strength = min(p.restore_strength, 0.72)
            reasons.append("off_angle_pose")

        # ── skin detail ─────────────────────────────────────────────────────
        if scores.skin_detail_score > 0.75:
            # Rich skin texture → use 1024 and moderate sharpening
            if p.restore_model == "auto":
                p.restore_model = self._pick_model(available_models, prefer_1024=True)
            p.sharpen_strength = max(p.sharpen_strength, 0.35)
        elif scores.skin_detail_score < 0.35:
            # Flat / smooth skin → protect existing texture
            p.texture_preserve_blend = max(p.texture_preserve_blend, 0.20)
            if p.restore_model == "auto":
                p.restore_model = self._pick_model(available_models, prefer_1024=False)

        # ── resolve "auto" model ─────────────────────────────────────────────
        if p.restore_model == "auto":
            prefer = scores.overall > self.QUALITY_HIGH
            p.restore_model = self._pick_model(available_models, prefer_1024=prefer)

        p.reason = " | ".join(reasons) if reasons else "nominal"
        return p, float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def _pick_model(available: List[str], prefer_1024: bool) -> str:
        """Pick best available GPEN model based on preference."""
        if not available or available == ["None"]:
            return "None"
        has_1024 = [m for m in available if "1024" in m and m != "None"]
        has_512  = [m for m in available if "512"  in m and m != "None"]
        non_none = [m for m in available if m != "None"]
        if prefer_1024 and has_1024:
            return has_1024[0]
        if has_512:
            return has_512[0]
        return non_none[0] if non_none else "None"


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Output Artifact Detector
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ArtifactScores:
    """All scores 0 → 1  (0 = clean / no artefact, 1 = severe artefact)."""

    plastic_skin_score: float = 0.0   # GPEN over-restoration → waxy look
    grain_score: float        = 0.0   # residual high-freq noise
    edge_seam_score: float    = 0.0   # hard seam at face boundary

    def worst(self) -> float:
        return max(self.plastic_skin_score, self.grain_score, self.edge_seam_score)

    def __repr__(self):
        return (
            f"Artifacts(plastic={self.plastic_skin_score:.2f}, "
            f"grain={self.grain_score:.2f}, seam={self.edge_seam_score:.2f})"
        )


class OutputArtifactDetector:
    """
    Detects post-swap artefacts by comparing restored face to swapped-only face.

    plastic_skin  : texture energy drop  (restored  <  raw swap)
    grain         : high-freq RMS in face region of restored image
    edge_seam     : gradient discontinuity at face mask boundary
    """

    PLASTIC_DROP_THRESH  = 0.25    # detail ratio drop  → starts counting as plastic
    PLASTIC_HEAVY_THRESH = 0.55    # detail ratio drop  → heavy plastic
    GRAIN_CLEAN_RMS      = 4.0
    GRAIN_NOISY_RMS      = 20.0
    SEAM_CLEAN_GRAD      = 8.0
    SEAM_HARD_GRAD       = 35.0

    def detect(self,
               swapped_img:  np.ndarray,
               restored_img: np.ndarray,
               face_mask:    Optional[np.ndarray] = None) -> ArtifactScores:
        """
        Args:
            swapped_img  : Image after face-swap, BEFORE restoration (BGR uint8).
            restored_img : Image after restoration (BGR uint8).
            face_mask    : Float mask [0-1] same HW as images (optional).
        """
        plastic = self._detect_plastic_skin(swapped_img, restored_img, face_mask)
        grain   = self._detect_grain(restored_img, face_mask)
        seam    = self._detect_seam(swapped_img, restored_img, face_mask)
        return ArtifactScores(
            plastic_skin_score = plastic,
            grain_score        = grain,
            edge_seam_score    = seam,
        )

    # ------------------------------------------------------------------
    def _texture_energy(self, img: np.ndarray, mask: Optional[np.ndarray]) -> float:
        gray = _to_gray(img).astype(np.float32)
        lap  = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        if mask is not None:
            valid = (mask > 0.25)
            return float(np.mean(lap[valid])) if np.any(valid) else float(np.mean(lap))
        return float(np.mean(lap))

    def _detect_plastic_skin(self, swapped: np.ndarray, restored: np.ndarray,
                              mask: Optional[np.ndarray]) -> float:
        e_before = self._texture_energy(swapped,  mask)
        e_after  = self._texture_energy(restored, mask)
        if e_before < 1e-6:
            return 0.0
        ratio = e_after / e_before   # <1 → smoothed out
        drop  = 1.0 - ratio
        if drop < self.PLASTIC_DROP_THRESH:
            return 0.0
        s = (drop - self.PLASTIC_DROP_THRESH) / max(
            1e-6, self.PLASTIC_HEAVY_THRESH - self.PLASTIC_DROP_THRESH)
        return float(np.clip(s, 0.0, 1.0))

    def _detect_grain(self, restored: np.ndarray, mask: Optional[np.ndarray]) -> float:
        gray      = _to_gray(restored).astype(np.float32)
        smoothed  = cv2.bilateralFilter(restored, d=5, sigmaColor=30, sigmaSpace=30)
        residual  = gray - _to_gray(smoothed).astype(np.float32)
        if mask is not None:
            valid = (mask > 0.25)
            rms = float(np.sqrt(np.mean(residual[valid] ** 2))) if np.any(valid) else float(np.sqrt(np.mean(residual ** 2)))
        else:
            rms = float(np.sqrt(np.mean(residual ** 2)))
        s = (rms - self.GRAIN_CLEAN_RMS) / max(1e-6, self.GRAIN_NOISY_RMS - self.GRAIN_CLEAN_RMS)
        return float(np.clip(s, 0.0, 1.0))

    def _detect_seam(self, swapped: np.ndarray, restored: np.ndarray,
                     mask: Optional[np.ndarray]) -> float:
        if mask is None:
            return 0.0
        # Build edge ring from mask
        m8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        k  = max(3, int(min(mask.shape[:2]) * 0.015))
        k  = k | 1  # ensure odd
        inner = cv2.erode( m8, np.ones((k, k), np.uint8), iterations=2)
        outer = cv2.dilate(m8, np.ones((k, k), np.uint8), iterations=2)
        ring  = (outer.astype(np.float32) - inner.astype(np.float32)) / 255.0
        ring  = np.clip(ring, 0, 1)

        if ring.sum() < 10:
            return 0.0

        diff = np.abs(
            restored.astype(np.float32) - swapped.astype(np.float32)
        ).mean(axis=2)

        ring_mean = float(np.sum(diff * ring) / np.sum(ring))
        s = (ring_mean - self.SEAM_CLEAN_GRAD) / max(1e-6, self.SEAM_HARD_GRAD - self.SEAM_CLEAN_GRAD)
        return float(np.clip(s, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Post-processing helpers  (colour match, denoise, sharpen, texture blend)
# ──────────────────────────────────────────────────────────────────────────────

def apply_color_match(source: np.ndarray, target: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Transfer luminance and colour statistics from target scene to result.
    Uses Lab colour space mean/std matching (Reinhard-style).

    Args:
        source   : Result image to recolour  (BGR uint8).
        target   : Reference image whose statistics we match  (BGR uint8).
        strength : Blend between original (0) and fully matched (1).
    """
    src_lab = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

    matched = src_lab.copy()
    for ch in range(3):
        s_mean, s_std = src_lab[:, :, ch].mean(), src_lab[:, :, ch].std() + 1e-6
        t_mean, t_std = tgt_lab[:, :, ch].mean(), tgt_lab[:, :, ch].std() + 1e-6
        matched[:, :, ch] = (src_lab[:, :, ch] - s_mean) * (t_std / s_std) + t_mean

    matched = np.clip(matched, 0, 255).astype(np.uint8)
    result  = cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)

    if strength < 1.0:
        result = cv2.addWeighted(result, strength, source, 1.0 - strength, 0)
    return result


def apply_denoise(img: np.ndarray, strength: float) -> np.ndarray:
    """Fast bilateral denoise scaled by strength (0–1)."""
    if strength <= 0.01:
        return img
    h_val = int(np.clip(strength * 12, 2, 12))
    return cv2.bilateralFilter(img, d=7, sigmaColor=h_val * 7, sigmaSpace=h_val * 7)


def apply_sharpen(img: np.ndarray, strength: float) -> np.ndarray:
    """Unsharp-mask sharpening scaled by strength (0–1)."""
    if strength <= 0.01:
        return img
    amount  = strength * 1.5
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def apply_texture_preserve(original: np.ndarray, result: np.ndarray,
                            blend: float, face_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Re-introduce high-frequency detail from `original` into `result`
    (counteracts GPEN over-smoothing / plastic skin).
    """
    if blend <= 0.01:
        return result

    orig_f   = original.astype(np.float32)
    result_f = result.astype(np.float32)

    # High-pass of original
    orig_lf   = cv2.GaussianBlur(orig_f, (0, 0), sigmaX=2.5)
    orig_hf   = orig_f - orig_lf
    blended_f = result_f + orig_hf * blend

    if face_mask is not None:
        m = face_mask[:, :, None]
        blended_f = blended_f * m + orig_f * (1.0 - m)

    return np.clip(blended_f, 0, 255).astype(np.uint8)


def apply_grain_dampen(img: np.ndarray, grain_score: float) -> np.ndarray:
    """
    Mild noise reduction when artifact detector flags grain.
    Uses fast NLMeans with conservative h parameter.
    """
    if grain_score < 0.25:
        return img
    h_val = int(np.clip(grain_score * 8, 2, 8))
    try:
        return cv2.fastNlMeansDenoisingColored(img, None, h_val, h_val, 7, 21)
    except Exception:
        return cv2.bilateralFilter(img, 7, sigmaColor=h_val * 6, sigmaSpace=h_val * 6)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Adaptive Pipeline Report
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptivePipelineReport:
    """Full audit trail of one adaptive run."""

    source_quality: FaceQualityScores = field(default_factory=FaceQualityScores)
    confidence:     float             = 1.0
    flagged_for_review: bool          = False

    passes: List[Dict[str, Any]] = field(default_factory=list)

    final_artifacts: ArtifactScores   = field(default_factory=ArtifactScores)
    success: bool                     = True
    message: str                      = ""

    def summary(self) -> str:
        lines = [
            f"══ ReActor V3 Adaptive Pipeline Report ══",
            f"  Source quality  : {self.source_quality}",
            f"  Confidence      : {self.confidence:.2f}",
            f"  Passes          : {len(self.passes)}",
        ]
        for i, p in enumerate(self.passes, 1):
            lines.append(
                f"  Pass {i}: params={p.get('params_reason','?')}  "
                f"artifacts={p.get('artifacts','?')}"
            )
        lines += [
            f"  Final artifacts : {self.final_artifacts}",
            f"  Flagged review  : {self.flagged_for_review}",
            f"  Status          : {'OK' if self.success else 'FAIL'} — {self.message}",
            "═══════════════════════════════════════",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Main Adaptive Pipeline
# ──────────────────────────────────────────────────────────────────────────────

CONFIDENCE_REVIEW_THRESHOLD = 0.30   # below → flag for manual review
MAX_RETRIES                 = 2      # maximum adaptive correction passes


class AdaptiveReActorPipeline:
    """
    Drop-in wrapper around ReActorV3 that applies the full adaptive feedback loop.

    Usage:
        engine   = get_reactor_v3_engine(models_path)
        pipeline = AdaptiveReActorPipeline(engine)
        result, report = pipeline.run(
            source_img, target_img,
            source_face_index=0,
            target_face_index=0,
            gender_match='S',
        )
    """

    def __init__(self, engine):
        """
        Args:
            engine : A ReActorV3 instance (from reactor_v3_swapper_new.py).
        """
        self.engine     = engine
        self.analyzer   = FaceQualityAnalyzer()
        self.selector   = AdaptiveParamSelector()
        self.detector   = OutputArtifactDetector()

    # ─────────────────────────────────────────────────────────
    def run(self,
            source_img:         np.ndarray,
            target_img:         np.ndarray,
            source_face_index:  int  = 0,
            target_face_index:  int  = 0,
            gender_match:       str  = 'S',
            force_params:       Optional[AdaptiveParams] = None
            ) -> Tuple[np.ndarray, AdaptivePipelineReport]:
        """
        Run adaptive face-swap pipeline.

        Returns:
            (result_image: BGR numpy, report: AdaptivePipelineReport)
        """
        report = AdaptivePipelineReport()

        # ── 1. Analyse source face quality ────────────────────────────────
        try:
            self.engine.initialize_face_analyser()
            src_faces = self.engine.get_faces(source_img)
            src_face  = src_faces[min(source_face_index, max(0, len(src_faces)-1))] \
                        if src_faces else None
        except Exception:
            src_face = None

        scores = self.analyzer.analyze(source_img, src_face)
        report.source_quality = scores

        print(f"[Adaptive] {scores}")

        # ── 2. Confidence gate ────────────────────────────────────────────
        available_models = self.engine.get_available_restorers()

        if force_params is not None:
            params     = force_params
            confidence = scores.overall
        else:
            params, confidence = self.selector.select(scores, available_models)

        report.confidence = confidence

        if confidence < CONFIDENCE_REVIEW_THRESHOLD:
            report.flagged_for_review = True
            report.success  = False
            report.message  = (
                f"Confidence {confidence:.2f} below threshold {CONFIDENCE_REVIEW_THRESHOLD}. "
                "Flagged for manual review — returning original target."
            )
            print(f"[Adaptive] ⚠  {report.message}")
            return target_img.copy(), report

        print(f"[Adaptive] Params (pass 1): {params}")

        # ── 3. First pass ─────────────────────────────────────────────────
        result, swapped_only, face_mask, pass_report = self._single_pass(
            source_img, target_img, src_face,
            source_face_index, target_face_index,
            gender_match, params
        )
        report.passes.append(pass_report)

        # ── 4. Artifact detection ─────────────────────────────────────────
        artifacts = self.detector.detect(swapped_only, result, face_mask)
        pass_report["artifacts"] = repr(artifacts)
        print(f"[Adaptive] Pass 1 artifacts: {artifacts}")

        # ── 5. Adaptive retry loop ────────────────────────────────────────
        for retry in range(MAX_RETRIES):
            if artifacts.worst() < 0.30:
                break   # clean enough

            corrected = self._correct_params(params, artifacts, scores)
            if corrected is None:
                print("[Adaptive] No further correction possible.")
                break

            params = corrected
            print(f"[Adaptive] Retry {retry+1}: {params}")

            result, swapped_only, face_mask, pass_report = self._single_pass(
                source_img, target_img, src_face,
                source_face_index, target_face_index,
                gender_match, params
            )
            artifacts = self.detector.detect(swapped_only, result, face_mask)
            pass_report["artifacts"] = repr(artifacts)
            report.passes.append(pass_report)
            print(f"[Adaptive] Pass {retry+2} artifacts: {artifacts}")

        # ── 6. Final artefact post-processing ────────────────────────────
        result = self._final_artifact_fix(result, swapped_only, artifacts, face_mask)

        report.final_artifacts = artifacts
        report.success  = True
        report.message  = (
            f"Completed {len(report.passes)} pass(es). "
            f"Final artifacts: worst={artifacts.worst():.2f}"
        )
        print(f"[Adaptive] Done. {report.message}")
        return result, report

    # ─────────────────────────────────────────────────────────
    def _single_pass(self,
                     source_img:        np.ndarray,
                     target_img:        np.ndarray,
                     src_face,
                     source_face_index: int,
                     target_face_index: int,
                     gender_match:      str,
                     params:            AdaptiveParams
                     ):
        """
        Execute one full swap + restore pass applying all AdaptiveParams.

        Returns:
            (result, swapped_only, face_mask, pass_report_dict)
        """
        pass_report: Dict[str, Any] = {"params_reason": params.reason}

        # Pre-process target: optional denoise
        working_target = apply_denoise(target_img.copy(), params.denoise_strength)

        # Colour match target to source before swap  (scene normalisation)
        if params.color_match:
            working_target = apply_color_match(
                working_target, source_img, strength=params.color_match_strength
            )

        # Swap (no restoration yet — capture raw swap for artifact comparison)
        swapped_only, _status = self.engine.process(
            source_img         = source_img,
            target_img         = working_target,
            source_face_index  = source_face_index,
            target_face_index  = target_face_index,
            restore_model      = None,          # raw swap, no restore
            gender_match       = gender_match,
        )

        # Build face mask for artefact detection (re-detect faces on swapped image)
        face_mask = self._build_face_mask(swapped_only, target_face_index)

        # If swap strength < 1.0, blend swapped back with target
        if params.swap_strength < 0.99:
            swapped_blended = cv2.addWeighted(
                swapped_only, params.swap_strength,
                working_target, 1.0 - params.swap_strength, 0
            )
        else:
            swapped_blended = swapped_only

        # Apply restoration
        restore_model = params.restore_model
        if restore_model and restore_model != "None":
            restored, _s = self.engine.process(
                source_img        = source_img,
                target_img        = working_target,
                source_face_index = source_face_index,
                target_face_index = target_face_index,
                restore_model     = restore_model,
                gender_match      = gender_match,
            )
            # If restore_strength < 1.0, blend with raw swap
            if params.restore_strength < 0.99:
                result = cv2.addWeighted(
                    restored,       params.restore_strength,
                    swapped_blended, 1.0 - params.restore_strength, 0
                )
            else:
                result = restored
        else:
            result = swapped_blended

        # Texture preservation blend (counteract GPEN over-smooting)
        if params.texture_preserve_blend > 0.01:
            result = apply_texture_preserve(
                working_target, result, params.texture_preserve_blend, face_mask
            )

        # Post-sharpening
        result = apply_sharpen(result, params.sharpen_strength)

        pass_report["restore_model"] = restore_model
        pass_report["restore_strength"] = params.restore_strength
        pass_report["swap_strength"] = params.swap_strength
        return result, swapped_only, face_mask, pass_report

    # ─────────────────────────────────────────────────────────
    def _correct_params(self,
                        prev_params: AdaptiveParams,
                        artifacts:   ArtifactScores,
                        scores:      FaceQualityScores
                        ) -> Optional[AdaptiveParams]:
        """
        Derive corrected AdaptiveParams from artifact scores.
        Returns None if no meaningful correction can be made.
        """
        import copy
        p = copy.copy(prev_params)
        changed = False

        if artifacts.plastic_skin_score > 0.30:
            # Over-restored → pull restoration back + add texture-preservation
            p.restore_strength       = max(0.30, p.restore_strength - 0.20)
            p.texture_preserve_blend = min(0.50, p.texture_preserve_blend + 0.20)
            p.sharpen_strength       = max(0.0,  p.sharpen_strength - 0.10)
            p.reason = f"retry:plastic(score={artifacts.plastic_skin_score:.2f})"
            changed = True

        if artifacts.grain_score > 0.30:
            # Noisy output → mild denoise, no extra sharpening
            p.denoise_strength = min(0.60, prev_params.denoise_strength + 0.20)
            p.sharpen_strength = max(0.0,  p.sharpen_strength - 0.15)
            p.reason += f"|retry:grain(score={artifacts.grain_score:.2f})"
            changed = True

        if artifacts.edge_seam_score > 0.35:
            # Hard boundary → reduce swap + restoration aggression
            p.swap_strength    = max(0.55, p.swap_strength    - 0.15)
            p.restore_strength = max(0.35, p.restore_strength - 0.10)
            p.reason += f"|retry:seam(score={artifacts.edge_seam_score:.2f})"
            changed = True

        return p if changed else None

    # ─────────────────────────────────────────────────────────
    def _final_artifact_fix(self,
                             result:      np.ndarray,
                             raw_swap:    np.ndarray,
                             artifacts:   ArtifactScores,
                             face_mask:   Optional[np.ndarray] = None
                             ) -> np.ndarray:
        """
        One-shot post-processing to clean up any remaining artefacts.
        """
        # Grain dampen
        if artifacts.grain_score > 0.25:
            result = apply_grain_dampen(result, artifacts.grain_score)

        # Plastic skin: final texture-preservation pass
        if artifacts.plastic_skin_score > 0.25:
            extra_blend = min(0.30, artifacts.plastic_skin_score * 0.40)
            result = apply_texture_preserve(raw_swap, result, extra_blend, face_mask)

        return result

    # ─────────────────────────────────────────────────────────
    def _build_face_mask(self, img: np.ndarray, face_index: int) -> Optional[np.ndarray]:
        """Detect faces and return a soft blended mask for the target face."""
        try:
            faces = self.engine.get_faces(img)
            if not faces:
                return None
            face = faces[min(face_index, len(faces) - 1)]
            mask, _ = self.engine._build_soft_face_mask(face, img.shape)
            return mask
        except Exception:
            return None
