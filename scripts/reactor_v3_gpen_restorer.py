"""
ReActor V3 - GPEN Restorer Module
High-fidelity blind face restoration using GPEN-BFR-512 and GPEN-BFR-1024

This module implements the GAN Prior Embedded Network (GPEN) for ultra-high quality
face restoration, supporting both 512x512 and 1024x1024 resolutions with proper
normalization and tensor handling.
"""

import cv2
import numpy as np
import onnxruntime
import os
from typing import Optional, Tuple

# Global cache to prevent reloading models on every frame
LOADED_GPEN_MODELS = {}


class GPENRestorer:
    """
    GPEN (GAN Prior Embedded Network) restoration wrapper for ReActor V3.
    
    This class handles:
    - Automatic resolution detection (512 or 1024)
    - Proper normalization (mean=0.5, std=0.5, range=[-1,1])
    - GPU acceleration via ONNX Runtime
    - Memory-efficient tensor operations
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize the GPEN restoration model.
        
        Args:
            model_path: Absolute path to the GPEN .onnx model file
            device: 'cuda' for GPU or 'cpu' for CPU execution
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GPEN model not found at: {model_path}")
        
        # Configure Execution Providers based on availability
        # Priority: CUDA > DirectML > CPU
        available_providers = onnxruntime.get_available_providers()
        
        if device == 'cuda' and 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print(f"[ReActor V3] Using CUDA acceleration for GPEN")
        elif 'DmlExecutionProvider' in available_providers:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            print(f"[ReActor V3] Using DirectML acceleration for GPEN")
        else:
            providers = ['CPUExecutionProvider']
            print(f"[ReActor V3] WARNING: Using CPU for GPEN (slow performance expected)")
        
        # Initialize the ONNX Runtime Inference Session
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.session = onnxruntime.InferenceSession(
                model_path, 
                sess_options=sess_options,
                providers=providers
            )
            
            # Verify which provider is actually being used
            active_providers = self.session.get_providers()
            print(f"[ReActor V3] GPEN active execution providers: {active_providers}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load GPEN model: {e}")
        
        # Introspect the model to determine input specifications
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        
        # Automatically detect resolution (Height is typically index 2 in NCHW format)
        # Expected shape: [1, 3, H, W] or ['batch', 3, H, W]
        if isinstance(input_shape[2], int):
            self.resolution = input_shape[2]
        else:
            # Dynamic shape - default to 512
            self.resolution = 512
            print(f"[ReActor V3] WARNING: Dynamic input shape detected, defaulting to 512")
        
        self.model_name = os.path.basename(model_path)
        print(f"[ReActor V3] Initialized GPEN model: {self.model_name} @ {self.resolution}x{self.resolution}")
        
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Prepare the image for GPEN inference with proper normalization.
        
        Critical normalization formula: (pixel/255.0 - 0.5) / 0.5
        This maps [0, 255] -> [-1.0, 1.0] as expected by StyleGAN-based models.
        
        Args:
            img: Numpy array (H, W, 3) in BGR format (OpenCV standard)
            
        Returns:
            tensor: Float32 array (1, 3, H, W) normalized to [-1, 1]
        """
        # 1. Resize to the model's native resolution (512 or 1024)
        if img.shape[0] != self.resolution or img.shape[1] != self.resolution:
            img = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        
        # 2. Convert BGR to RGB (CRITICAL: GPEN expects RGB order)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. Normalization Logic: (Pixel / 255.0 - 0.5) / 0.5
        # This maps [0, 255] -> [-1.0, 1.0]
        # Black (0) -> -1.0, Grey (127.5) -> 0.0, White (255) -> 1.0
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        
        # 4. Transpose from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
        img = np.transpose(img, (2, 0, 1))
        
        # 5. Add Batch Dimension: (3, H, W) -> (1, 3, H, W)
        img = np.expand_dims(img, axis=0)
        
        return img.astype(np.float32)
    
    def postprocess(self, output_tensor: np.ndarray) -> np.ndarray:
        """
        Convert the raw model output back to a usable BGR image.
        
        Args:
            output_tensor: Float32 array (1, 3, H, W) in range ~[-1, 1]
            
        Returns:
            image: Uint8 array (H, W, 3) in BGR format
        """
        # 1. Remove batch dimension and transpose back to HWC
        output = np.squeeze(output_tensor)  # (3, H, W)
        output = np.transpose(output, (1, 2, 0))  # (H, W, 3)
        
        # 2. Clip values to valid range [-1, 1] to prevent integer overflow
        # The network may output slightly outside this range due to floating point ops
        output = np.clip(output, -1.0, 1.0)
        
        # 3. Denormalize: reverse the preprocessing
        # (Value + 1) / 2 * 255 maps [-1, 1] -> [0, 255]
        output = (output + 1.0) / 2.0
        output = output * 255.0
        
        # 4. Round and cast to 8-bit unsigned integer
        output = np.round(output).astype(np.uint8)
        
        # 5. Convert RGB back to BGR for OpenCV compatibility
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        return output
    
    def restore(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Main entry point for face restoration.
        
        Args:
            face_image: BGR image (H, W, 3) containing a face
            
        Returns:
            Restored BGR image, or None if restoration fails
        """
        if face_image is None or face_image.size == 0:
            print("[ReActor V3] ERROR: Empty face image provided to GPEN")
            return None
        
        try:
            print(f"[ReActor V3 GPEN] Input face shape: {face_image.shape}, dtype: {face_image.dtype}")
            
            # Preprocess: normalize and prepare tensor
            input_tensor = self.preprocess(face_image)
            
            print(f"[ReActor V3 GPEN] Preprocessed tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            print(f"[ReActor V3 GPEN] Tensor value range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
            
            # Run ONNX inference
            result = self.session.run(
                [self.output_name], 
                {self.input_name: input_tensor}
            )
            
            print(f"[ReActor V3 GPEN] Inference complete - output shape: {result[0].shape}")
            
            # Postprocess: denormalize and convert back to image
            restored_face = self.postprocess(result[0])
            
            print(f"[ReActor V3 GPEN] Postprocessed face shape: {restored_face.shape}, dtype: {restored_face.dtype}")
            
            return restored_face
            
        except Exception as e:
            print(f"[ReActor V3] ERROR during GPEN restoration: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def __del__(self):
        """Cleanup ONNX session on deletion"""
        if hasattr(self, 'session'):
            del self.session


def get_gpen_restorer(model_path: str, device: str = 'cuda') -> GPENRestorer:
    """
    Factory function to get or create a GPEN restorer with caching.
    
    This prevents reloading the heavy models on every frame, which would
    cause massive performance degradation (especially for 1024 models).
    
    Args:
        model_path: Path to the GPEN .onnx model
        device: 'cuda' or 'cpu'
        
    Returns:
        Cached or new GPENRestorer instance
    """
    cache_key = f"{model_path}_{device}"
    
    if cache_key not in LOADED_GPEN_MODELS:
        print(f"[ReActor V3] Loading GPEN model into cache: {os.path.basename(model_path)}")
        LOADED_GPEN_MODELS[cache_key] = GPENRestorer(model_path, device)
    
    return LOADED_GPEN_MODELS[cache_key]


def clear_gpen_cache():
    """Clear all cached GPEN models to free VRAM"""
    global LOADED_GPEN_MODELS
    for key in list(LOADED_GPEN_MODELS.keys()):
        del LOADED_GPEN_MODELS[key]
    LOADED_GPEN_MODELS.clear()
    print("[ReActor V3] GPEN model cache cleared")


def get_available_gpen_models(models_dir: str) -> list:
    """
    Scan the models directory for available GPEN .onnx files.
    
    Args:
        models_dir: Path to the facerestore_models directory
        
    Returns:
        List of model filenames
    """
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for file in os.listdir(models_dir):
        if file.lower().endswith('.onnx') and 'gpen' in file.lower():
            models.append(file)
    
    return sorted(models)
