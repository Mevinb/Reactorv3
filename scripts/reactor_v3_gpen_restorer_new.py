"""
ReActor V3 - GPEN Face Restoration Module

This module provides GPEN-based face restoration using WebUI's FaceRestoreHelper,
exactly like GFPGAN and CodeFormer do.
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from typing import Optional
import torch

# Setup cuDNN path for ONNX Runtime CUDA provider
def setup_cudnn_path():
    """Add cuDNN and cuBLAS to PATH if available"""
    try:
        import site
        site_packages = site.getsitepackages()
        paths_added = []
        for site_pkg in site_packages:
            # Add cuDNN path
            cudnn_bin_path = os.path.join(site_pkg, 'nvidia', 'cudnn', 'bin')
            if os.path.exists(cudnn_bin_path):
                current_path = os.environ.get('PATH', '')
                if cudnn_bin_path not in current_path:
                    os.environ['PATH'] = cudnn_bin_path + os.pathsep + current_path
                    paths_added.append(cudnn_bin_path)
            
            # Add cuBLAS path
            cublas_bin_path = os.path.join(site_pkg, 'nvidia', 'cublas', 'bin')
            if os.path.exists(cublas_bin_path):
                current_path = os.environ.get('PATH', '')
                if cublas_bin_path not in current_path:
                    os.environ['PATH'] = cublas_bin_path + os.pathsep + current_path
                    paths_added.append(cublas_bin_path)
        
        if paths_added:
            print(f"[ReActor V3] Added CUDA libraries to PATH: {paths_added}")
            return True
        else:
            print("[ReActor V3] CUDA libraries (cuDNN/cuBLAS) not found in site-packages")
            return False
    except Exception as e:
        print(f"[ReActor V3] Error setting up CUDA libraries path: {e}")
        return False

# Setup cuDNN path on import
setup_cudnn_path()

# Use WebUI's face restoration infrastructure
import sys
webui_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, webui_path)

from modules import face_restoration_utils, devices


class GPENFaceRestorer:
    """
    GPEN face restoration using WebUI's FaceRestoreHelper infrastructure.
    This works exactly like GFPGAN and CodeFormer.
    """
    
    def __init__(self, model_path: str, resolution: int = 512, device: str = 'cuda'):
        """
        Initialize GPEN restorer.
        
        Args:
            model_path: Path to GPEN ONNX model
            resolution: Model resolution (512 or 1024)
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.resolution = resolution
        self.device = device
        self.session = None
        self.face_helper = None
        
        self._initialize_model()
        self._initialize_face_helper()
    
    def _initialize_model(self):
        """Initialize ONNX Runtime session for GPEN"""
        # Suppress ONNX Runtime warnings about graph optimizations
        ort.set_default_logger_severity(3)  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
        
        providers = []
        if self.device == 'cuda':
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }))
        providers.append('CPUExecutionProvider')
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3  # Suppress warnings
        
        print(f"[ReActor V3] GPEN ── Initializing ONNX Session ──")
        print(f"[ReActor V3] GPEN   Model: {os.path.basename(self.model_path)}")
        print(f"[ReActor V3] GPEN   Resolution: {self.resolution}x{self.resolution}")
        print(f"[ReActor V3] GPEN   Device: {self.device}")
        print(f"[ReActor V3] GPEN   Requested providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
        model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024) if os.path.exists(self.model_path) else 0
        print(f"[ReActor V3] GPEN   Model file size: {model_size_mb:.1f} MB")
        
        t0 = time.time()
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
        elapsed = time.time() - t0
        
        # Verify which provider is actually being used
        active_providers = self.session.get_providers()
        print(f"[ReActor V3] GPEN   Active execution providers: {active_providers}")
        print(f"[ReActor V3] GPEN   Session created in {elapsed:.2f}s")
        
        # Log input/output details
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        for inp in inputs:
            print(f"[ReActor V3] GPEN   Input: name='{inp.name}', shape={inp.shape}, type={inp.type}")
        for out in outputs:
            print(f"[ReActor V3] GPEN   Output: name='{out.name}', shape={out.shape}, type={out.type}")
        
        print(f"[ReActor V3] GPEN   Initialized: {os.path.basename(self.model_path)} @ {self.resolution}x{self.resolution}")
    
    def _initialize_face_helper(self):
        """Initialize WebUI's FaceRestoreHelper with correct resolution"""
        print(f"[ReActor V3] GPEN ── Initializing FaceRestoreHelper ──")
        device_torch = devices.device_codeformer if self.device == 'cuda' else devices.cpu
        print(f"[ReActor V3] GPEN   Device: {device_torch}")
        
        # Create FaceRestoreHelper with the correct face_size for this model
        from facexlib.detection import retinaface
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        
        if hasattr(retinaface, 'device'):
            retinaface.device = device_torch
        
        print(f"[ReActor V3] GPEN   Face size: {self.resolution}")
        print(f"[ReActor V3] GPEN   Detection model: retinaface_resnet50")
        print(f"[ReActor V3] GPEN   Use parse: True")
        
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=self.resolution,  # Use model's resolution (512 or 1024)
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=device_torch,
        )
        print(f"[ReActor V3] GPEN   FaceRestoreHelper initialized")
    
    def restore_with_gpen(self, cropped_face_t: torch.Tensor) -> torch.Tensor:
        """
        Restore a single cropped face tensor.
        This is called by FaceRestoreHelper for each detected face.
        
        Args:
            cropped_face_t: Normalized face tensor (1, 3, 512, 512) in range [-1, 1]
            
        Returns:
            Restored face tensor in range [-1, 1]
        """
        t0 = time.time()
        # Convert to numpy for ONNX
        face_np = cropped_face_t.cpu().numpy()
        input_shape = face_np.shape
        input_range = (float(np.min(face_np)), float(np.max(face_np)))
        print(f"[ReActor V3] GPEN   Processing cropped face: shape={input_shape}, range=[{input_range[0]:.3f}, {input_range[1]:.3f}]")
        
        # GPEN expects input in [-1, 1] range (already normalized by face_helper)
        # Resize if needed
        resized = False
        if face_np.shape[2] != self.resolution or face_np.shape[3] != self.resolution:
            # Convert to HWC for resize
            face_hwc = face_np.transpose(0, 2, 3, 1)[0]
            face_hwc = cv2.resize(face_hwc, (self.resolution, self.resolution))
            face_np = face_hwc.transpose(2, 0, 1)[np.newaxis, ...]
            resized = True
            print(f"[ReActor V3] GPEN   Resized input to {self.resolution}x{self.resolution}")
        
        # Run GPEN inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        t_infer = time.time()
        restored_np = self.session.run([output_name], {input_name: face_np.astype(np.float32)})[0]
        infer_elapsed = time.time() - t_infer
        
        output_range_before_clip = (float(np.min(restored_np)), float(np.max(restored_np)))
        print(f"[ReActor V3] GPEN   Inference completed in {infer_elapsed:.3f}s")
        print(f"[ReActor V3] GPEN   Output range before clip: [{output_range_before_clip[0]:.3f}, {output_range_before_clip[1]:.3f}]")
        
        # Step 7: Soft tanh compression instead of hard clip — prevents banding artifacts
        restored_np = np.tanh(restored_np)
        
        output_range_after_clip = (float(np.min(restored_np)), float(np.max(restored_np)))
        if output_range_before_clip[0] < -1.0 or output_range_before_clip[1] > 1.0:
            print(f"[ReActor V3] GPEN   ⚠ Values soft-compressed with tanh. After: [{output_range_after_clip[0]:.3f}, {output_range_after_clip[1]:.3f}]")
        
        # Compute face difference metrics
        diff = np.abs(restored_np - face_np)
        mean_change = float(np.mean(diff))
        max_change = float(np.max(diff))
        print(f"[ReActor V3] GPEN   Restoration change: mean={mean_change:.4f}, max={max_change:.4f}")
        
        # Convert back to torch tensor
        restored_t = torch.from_numpy(restored_np).to(cropped_face_t.device)
        
        total_elapsed = time.time() - t0
        print(f"[ReActor V3] GPEN   Total face restore time: {total_elapsed:.3f}s")
        
        return restored_t
    
    def enhance_face_region(self, full_img: np.ndarray, face_bbox) -> np.ndarray:
        """
        High-frequency detail enhancer for a single face region.

        Replaces full-face GPEN replacement with HF delta injection:
          Crop face → Upscale 512 → GPEN → Extract HF delta → Inject into original
          → Downscale → Feathered composite

        GPEN output is NEVER blended directly.  Only its additional high-frequency
        detail (relative to the swapped face) is injected back.

        Args:
            full_img:  Full BGR uint8 image.
            face_bbox: [x1, y1, x2, y2] bounding box of the swapped face.

        Returns:
            Enhanced full image in BGR uint8 (same resolution as input).
        """
        t0 = time.time()
        img_h, img_w = full_img.shape[:2]

        # Parse + clamp bbox
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        bbox_w, bbox_h = x2 - x1, y2 - y1

        if bbox_w < 8 or bbox_h < 8:
            print(f"[ReActor V3] GPEN HF: bbox too small ({bbox_w}x{bbox_h}), skipping")
            return full_img

        # ── Step 1: Crop face region ────────────────────────────────────
        face_crop = full_img[y1:y2, x1:x2].copy()

        # ── Step 2: Upscale to 512x512 with Lanczos (improves GPEN response) ──
        face_512 = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        # ── Step 3: Convert to float32 [0, 1] ──────────────────────────
        face_f32 = face_512.astype(np.float32) / 255.0

        # Prepare GPEN input: BGR→RGB, [0,1]→[-1,1], HWC→NCHW
        face_rgb  = face_f32[:, :, ::-1].copy()
        face_norm = face_rgb * 2.0 - 1.0

        if self.resolution != 512:
            face_norm_r = cv2.resize(face_norm, (self.resolution, self.resolution),
                                     interpolation=cv2.INTER_LANCZOS4)
        else:
            face_norm_r = face_norm

        face_nchw = face_norm_r.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        # ── Step 4 (part of Step 3): Run GPEN inference ─────────────────
        input_name  = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        t_infer = time.time()
        gpen_out_nchw = self.session.run(
            [output_name], {input_name: face_nchw}
        )[0]
        infer_elapsed = time.time() - t_infer
        print(f"[ReActor V3] GPEN HF inference: {infer_elapsed:.3f}s")

        # Convert GPEN output back to BGR float32 [0, 1]
        # NCHW → HWC, RGB → BGR, [-1,1] → [0,1]
        gpen_hwc  = gpen_out_nchw[0].transpose(1, 2, 0)
        gpen_bgr  = gpen_hwc[:, :, ::-1].copy()
        gpen_f32  = (gpen_bgr + 1.0) * 0.5

        # Ensure both arrays are 512x512 (model may output different res)
        if gpen_f32.shape[:2] != (512, 512):
            gpen_f32 = cv2.resize(gpen_f32, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        # ── Convert sRGB → linear space for unbiased HF arithmetic ─────
        # Gamma-corrected sRGB biases HF separation; linear space is correct.
        face_lin = np.power(np.clip(face_f32, 0.0, 1.0), 2.2)
        gpen_lin = np.power(np.clip(gpen_f32, 0.0, 1.0), 2.2)

        # ── Steps 4-5: Extract Low-Frequency layers ─────────────────────
        # sigma_lf=2.8: captures the 1-3px perceptual-clarity band (the range
        # that makes a face look sharp vs. AI-blurry). sigma=1.4 was too fine —
        # it only caught sub-pixel noise, missing inswapper's 128px softness.
        sigma_lf    = 2.8
        low_swapped = cv2.GaussianBlur(face_lin, (0, 0), sigmaX=sigma_lf)
        low_gpen    = cv2.GaussianBlur(gpen_lin, (0, 0), sigmaX=sigma_lf)

        # ── Step 5: Extract High-Frequency components ───────────────────
        hf_swapped = face_lin - low_swapped
        hf_gpen    = gpen_lin - low_gpen

        # ── Step 6: Compute HF delta (GPEN's added micro-detail only) ───
        hf_delta = hf_gpen - hf_swapped

        # ── Step 6b: Spatial normalization of HF delta ──────────────────
        # Prevents center over-sharpening and cheek under-sharpening.
        hf_strength  = np.mean(np.abs(hf_delta), axis=2, keepdims=True)
        hf_norm      = hf_delta / (hf_strength + 1e-6)
        target_strength = np.mean(hf_strength)
        hf_balanced  = hf_norm * target_strength

        # ── Step 7: Adaptive alpha injection ────────────────────────────
        # Alpha scales with how much extra HF GPEN actually added.
        # High ratio (face already has strong HF) → lower alpha.
        # Prevents over-injection on already-decent faces.
        hf_energy_face = float(np.mean(np.abs(hf_swapped)))
        hf_energy_gpen = float(np.mean(np.abs(hf_gpen)))
        ratio          = hf_energy_face / (hf_energy_gpen + 1e-6)
        # No floor: if face already has equal/more HF than GPEN, inject nothing.
        # Ceiling raised to 0.55 — blurry inswapper faces need more injection.
        alpha          = float(np.clip(0.55 * (1.0 - ratio), 0.0, 0.55))
        print(
            f"[ReActor V3] GPEN HF adaptive alpha: "
            f"hf_face={hf_energy_face:.5f}, hf_gpen={hf_energy_gpen:.5f}, "
            f"ratio={ratio:.3f} → alpha={alpha:.3f}"
        )

        enhanced_lin = face_lin + alpha * hf_balanced

        # ── Convert linear → sRGB ────────────────────────────────────────
        enhanced_lin = np.clip(enhanced_lin, 0.0, 1.0)
        enhanced_512 = np.power(enhanced_lin, 1.0 / 2.2)
        enhanced_512 = np.clip(enhanced_512, 0.0, 1.0).astype(np.float32)

        # ── Mild single-pass clarity boost (mid-frequency only) ──────────
        # Recovers the 2-5px clarity lost at the inswapper 128px bottleneck.
        # Conservative multipliers (1.15 mid, no high boost) — no halos.
        _g1 = cv2.GaussianBlur(enhanced_512, (0, 0), 1.2)   # fine-low
        _g2 = cv2.GaussianBlur(enhanced_512, (0, 0), 3.0)   # coarse-low
        _mid = _g1 - _g2                                     # clarity band
        enhanced_512 = np.clip(enhanced_512 + _mid * 0.30, 0.0, 1.0)

        # ── Step 9: Downscale back to original bbox size ─────────────────
        enhanced_face = cv2.resize(enhanced_512, (bbox_w, bbox_h),
                                   interpolation=cv2.INTER_AREA)
        enhanced_u8   = np.clip(enhanced_face * 255.0, 0, 255).astype(np.uint8)

        # ── Step 10: Feathered composite back to target image ────────────
        result = full_img.copy()

        mask   = np.zeros((bbox_h, bbox_w), dtype=np.float32)
        cx, cy = bbox_w // 2, bbox_h // 2
        ax     = max(2, int(bbox_w * 0.45))
        ay     = max(2, int(bbox_h * 0.45))
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
        # Scale feather to 15% of bbox for smoother compositing.
        feather = max(7, int(min(bbox_w, bbox_h) * 0.15))
        if feather % 2 == 0:
            feather += 1
        mask = cv2.GaussianBlur(mask, (feather, feather), feather * 0.35)
        mask = np.clip(mask, 0.0, 1.0)

        region   = result[y1:y2, x1:x2].astype(np.float32)
        mask_3ch = mask[:, :, None]
        blended  = enhanced_u8.astype(np.float32) * mask_3ch + region * (1.0 - mask_3ch)
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        elapsed = time.time() - t0
        print(
            f"[ReActor V3] GPEN HF enhancement: bbox=({x1},{y1},{x2},{y2}) "
            f"[{bbox_w}x{bbox_h}] → 512 → [{bbox_w}x{bbox_h}], {elapsed:.3f}s"
        )
        return result

    def restore(self, np_image: np.ndarray) -> np.ndarray:
        """
        Restore faces in an image using WebUI's FaceRestoreHelper.
        This is the main entry point, exactly like GFPGAN.restore()
        
        Args:
            np_image: Input image in BGR format (H, W, 3) uint8
            
        Returns:
            Restored image in BGR format
        """
        h, w = np_image.shape[:2]
        print(f"[ReActor V3] GPEN ── Full Image Restore ──")
        print(f"[ReActor V3] GPEN   Input image: {w}x{h} ({np_image.dtype})")
        print(f"[ReActor V3] GPEN   Model: {os.path.basename(self.model_path)} ({self.resolution}x{self.resolution})")
        
        # Analyze input image color stats
        mean_bgr = np.mean(np_image, axis=(0,1))
        print(f"[ReActor V3] GPEN   Input mean color (BGR): [{mean_bgr[0]:.1f}, {mean_bgr[1]:.1f}, {mean_bgr[2]:.1f}]")
        
        t0 = time.time()
        result = face_restoration_utils.restore_with_face_helper(
            np_image,
            self.face_helper,
            self.restore_with_gpen
        )
        elapsed = time.time() - t0
        
        # Analyze output
        mean_bgr_out = np.mean(result, axis=(0,1))
        color_shift = np.abs(mean_bgr_out - mean_bgr)
        print(f"[ReActor V3] GPEN   Output mean color (BGR): [{mean_bgr_out[0]:.1f}, {mean_bgr_out[1]:.1f}, {mean_bgr_out[2]:.1f}]")
        print(f"[ReActor V3] GPEN   Global color shift (BGR): [{color_shift[0]:.1f}, {color_shift[1]:.1f}, {color_shift[2]:.1f}]")
        print(f"[ReActor V3] GPEN   Full restore completed in {elapsed:.3f}s")
        
        return result


# Cache for loaded models
LOADED_GPEN_MODELS = {}


def get_gpen_restorer(model_path: str, device: str = 'cuda') -> GPENFaceRestorer:
    """
    Get or create a GPEN restorer instance.
    Models are cached to avoid reloading.
    
    Args:
        model_path: Path to GPEN ONNX model
        device: Device to use
        
    Returns:
        GPENFaceRestorer instance
    """
    cache_key = f"{model_path}_{device}"
    
    if cache_key not in LOADED_GPEN_MODELS:
        # Determine resolution from model name
        model_name = os.path.basename(model_path)
        if '1024' in model_name:
            resolution = 1024
        elif '512' in model_name:
            resolution = 512
        else:
            resolution = 512  # Default
        
        LOADED_GPEN_MODELS[cache_key] = GPENFaceRestorer(model_path, resolution, device)
        print(f"[ReActor V3] Loading GPEN model into cache: {model_name}")
    
    return LOADED_GPEN_MODELS[cache_key]


def clear_gpen_cache():
    """Clear all cached GPEN models to free VRAM"""
    global LOADED_GPEN_MODELS
    count = len(LOADED_GPEN_MODELS)
    for key in list(LOADED_GPEN_MODELS.keys()):
        del LOADED_GPEN_MODELS[key]
    LOADED_GPEN_MODELS.clear()
    print(f"[ReActor V3] GPEN model cache cleared ({count} model(s) released)")


def get_available_gpen_models(models_dir: str) -> list:
    """
    Get list of available GPEN models in the directory.
    
    Args:
        models_dir: Directory containing GPEN models
        
    Returns:
        List of model filenames
    """
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('.onnx') and 'GPEN' in file.upper():
            models.append(file)
    
    return sorted(models)
