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
