"""
ReActor V3 - Main Face Swapping Pipeline

This module orchestrates the complete face swapping workflow:
1. Face detection in both source and target images
2. Face swapping using InsightFace models
3. High-fidelity restoration using GPEN-512/1024
4. Seamless blending back into the original image
"""

import cv2
import numpy as np
import os
from typing import Optional, List, Tuple
import traceback
import torch
import gc

try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import model_zoo
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[ReActor V3] WARNING: InsightFace not installed. Please install: pip install insightface")

# Import memory management from WebUI Forge backend
import sys
_ext_scripts_path = os.path.dirname(os.path.abspath(__file__))
if _ext_scripts_path not in sys.path:
    sys.path.insert(0, _ext_scripts_path)

try:
    webui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if webui_path not in sys.path:
        sys.path.insert(0, webui_path)
    from backend import memory_management
    MEMORY_MANAGEMENT_AVAILABLE = True
except ImportError:
    MEMORY_MANAGEMENT_AVAILABLE = False
    print("[ReActor V3] WARNING: WebUI memory management not available, using basic cleanup")

from reactor_v3_gpen_restorer import get_gpen_restorer, get_available_gpen_models, clear_gpen_cache
from reactor_v3_face_utils import FaceAlignment


class ReActorV3:
    """
    Main ReActor V3 face swapping engine with GPEN integration.
    
    This is a completely independent implementation that doesn't interfere
    with existing reactor extensions.
    """
    
    def __init__(self, models_path: str):
        """
        Initialize ReActor V3 engine.
        
        Args:
            models_path: Root path for models directory
        """
        self.models_path = models_path
        
        # The models_path passed in is already the webui/models directory
        # So we can use it directly as the shared models root
        self.shared_models_root = models_path
        
        # Primary paths - use shared models directory
        self.insightface_path = os.path.join(self.shared_models_root, 'insightface')
        self.facerestore_path = os.path.join(self.shared_models_root, 'facerestore_models')
        
        # Create directories if they don't exist
        os.makedirs(self.insightface_path, exist_ok=True)
        os.makedirs(self.facerestore_path, exist_ok=True)
        
        # InsightFace components
        self.face_analyser = None
        self.face_swapper = None
        
        # GPEN restorer (loaded on demand)
        self.current_restorer = None
        self.current_restorer_name = None
        
        # Face alignment manager
        self.face_aligner = FaceAlignment(default_resolution=512)
        
        # Auto cleanup settings
        self.auto_cleanup = True  # Automatically free VRAM after processing
        
        print(f"[ReActor V3] Initialized with models path: {models_path}")
        print(f"[ReActor V3] Shared models root: {self.shared_models_root}")
        print(f"[ReActor V3] GPEN models path: {self.facerestore_path}")
        print(f"[ReActor V3] InsightFace path: {self.insightface_path}")
        print(f"[ReActor V3] Auto VRAM cleanup: {self.auto_cleanup}")
    
    def initialize_face_analyser(self):
        """
        Initialize InsightFace face detection and analysis.
        
        This uses the buffalo_l model by default for high accuracy.
        """
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace is not installed")
        
        if self.face_analyser is None:
            try:
                print("[ReActor V3] Initializing face analyzer...")
                self.face_analyser = FaceAnalysis(
                    name='buffalo_l',
                    root=self.insightface_path,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
                print("[ReActor V3] Face analyzer ready")
            except Exception as e:
                print(f"[ReActor V3] ERROR initializing face analyzer: {e}")
                print("[ReActor V3] Please download InsightFace models:")
                print("           They will be auto-downloaded on first use")
                raise
    
    def initialize_face_swapper(self, model_name: str = 'inswapper_128.onnx'):
        """
        Initialize the face swapping model.
        
        Args:
            model_name: Name of the swapper model to use
        """
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace is not installed")
        
        if self.face_swapper is None:
            try:
                print(f"[ReActor V3] Initializing face swapper: {model_name}")
                
                # Look for the model in standard WebUI shared locations
                model_path = None
                search_paths = [
                    # Shared WebUI models directory (primary)
                    os.path.join(self.insightface_path, model_name),  # webui/models/insightface/
                    os.path.join(self.insightface_path, 'models', model_name),  # webui/models/insightface/models/
                    os.path.join(self.shared_models_root, 'reactor', 'faces', model_name),
                    os.path.join(self.shared_models_root, 'reactor', model_name),
                    os.path.join(self.shared_models_root, 'hyperswap', model_name),
                    # Legacy extension-specific paths (fallback)
                    os.path.join(self.models_path, 'insightface', 'models', model_name),
                    os.path.join(self.models_path, model_name),
                    os.path.join(self.models_path, 'reactor', model_name),
                ]
                
                for path in search_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                
                if model_path is None:
                    # Show only the most relevant paths to user
                    user_friendly_paths = [
                        os.path.join(self.insightface_path, model_name),
                        os.path.join(self.insightface_path, 'models', model_name),
                        os.path.join(self.shared_models_root, 'reactor', model_name),
                        os.path.join(self.shared_models_root, 'hyperswap', model_name),
                    ]
                    raise FileNotFoundError(
                        f"Face swapper model not found: {model_name}\n"
                        f"Please place it in the shared WebUI models directory:\n" + 
                        "\n".join(user_friendly_paths) +
                        f"\n\nNote: ReActor V3 uses the same models as other reactor extensions."
                    )
                
                self.face_swapper = model_zoo.get_model(
                    model_path,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                print(f"[ReActor V3] Face swapper ready: {model_path}")
                
            except Exception as e:
                print(f"[ReActor V3] ERROR initializing face swapper: {e}")
                raise
    
    def get_faces(self, img: np.ndarray) -> List:
        """
        Detect all faces in an image.
        
        Args:
            img: BGR image from OpenCV
            
        Returns:
            List of face detection objects
        """
        if self.face_analyser is None:
            self.initialize_face_analyser()
        
        try:
            faces = self.face_analyser.get(img)
            return faces
        except Exception as e:
            print(f"[ReActor V3] Error detecting faces: {e}")
            return []
    
    def swap_face(self, source_face, target_face, target_img: np.ndarray) -> np.ndarray:
        """
        Perform the actual face swap operation.
        
        Args:
            source_face: Source face detection object
            target_face: Target face detection object
            target_img: Target image
            
        Returns:
            Image with swapped face
        """
        if self.face_swapper is None:
            self.initialize_face_swapper()
        
        try:
            # Keep paste_back=True for the simple swap case
            result = self.face_swapper.get(target_img, target_face, source_face, paste_back=True)
            return result
        except Exception as e:
            print(f"[ReActor V3] Error swapping face: {e}")
            return target_img
    
    def load_restorer(self, model_name: str) -> bool:
        """
        Load a GPEN restoration model.
        
        Args:
            model_name: Name of the GPEN model file
            
        Returns:
            True if successful
        """
        if model_name == self.current_restorer_name and self.current_restorer is not None:
            return True  # Already loaded
        
        if not model_name or model_name.lower() == 'none':
            self.current_restorer = None
            self.current_restorer_name = None
            return True
        
        # Use only the standard WebUI facerestore_models directory
        model_path = os.path.join(self.facerestore_path, model_name)
        
        if not os.path.exists(model_path):
            print(f"[ReActor V3] ERROR: Model not found: {model_name}")
            print(f"[ReActor V3] Place GPEN models in: {self.facerestore_path}")
            return False
        
        try:
            self.current_restorer = get_gpen_restorer(model_path, device='cuda')
            self.current_restorer_name = model_name
            return True
        except Exception as e:
            print(f"[ReActor V3] ERROR loading restorer: {e}")
            return False
    
    def restore_face(self, face_img: np.ndarray, resolution: int = 512) -> Optional[np.ndarray]:
        """
        Restore a face using the current GPEN model.
        
        Args:
            face_img: Aligned face image
            resolution: Expected resolution (512 or 1024)
            
        Returns:
            Restored face image or None
        """
        if self.current_restorer is None:
            return face_img  # No restoration
        
        try:
            # Ensure the face is at the correct resolution for the model
            if face_img.shape[0] != self.current_restorer.resolution or \
               face_img.shape[1] != self.current_restorer.resolution:
                face_img = cv2.resize(
                    face_img, 
                    (self.current_restorer.resolution, self.current_restorer.resolution),
                    interpolation=cv2.INTER_LINEAR
                )
            
            restored = self.current_restorer.restore(face_img)
            return restored if restored is not None else face_img
            
        except Exception as e:
            print(f"[ReActor V3] Error restoring face: {e}")
            return face_img
    
    def process(self, 
                source_img: np.ndarray,
                target_img: np.ndarray,
                source_face_index: int = 0,
                target_face_index: int = 0,
                restore_model: str = None,
                auto_resolution: bool = True) -> Tuple[np.ndarray, str]:
        """
        Complete face swapping pipeline with GPEN restoration.
        
        Args:
            source_img: Source image containing the face to copy
            target_img: Target image where face will be swapped
            source_face_index: Which face to use from source (0 = first)
            target_face_index: Which face to replace in target (0 = first)
            restore_model: GPEN model name to use for restoration
            auto_resolution: Automatically select 512/1024 based on face size
            
        Returns:
            (result_image, status_message) tuple
        """
        if source_img is None or target_img is None:
            return target_img, "Error: Missing source or target image"
        
        try:
            # Step 1: Initialize components
            if self.face_analyser is None:
                self.initialize_face_analyser()
            
            if self.face_swapper is None:
                self.initialize_face_swapper()
            
            # Step 2: Load restoration model if specified
            if restore_model:
                if not self.load_restorer(restore_model):
                    return target_img, f"Error: Failed to load restoration model: {restore_model}"
            
            # Step 3: Detect faces
            print(f"[ReActor V3] Detecting faces...")
            source_faces = self.get_faces(source_img)
            target_faces = self.get_faces(target_img)
            
            if len(source_faces) == 0:
                return target_img, "Error: No face detected in source image"
            
            if len(target_faces) == 0:
                return target_img, "Error: No face detected in target image"
            
            # Select faces by index
            if source_face_index >= len(source_faces):
                source_face_index = 0
            if target_face_index >= len(target_faces):
                target_face_index = 0
            
            source_face = source_faces[source_face_index]
            target_face = target_faces[target_face_index]
            
            print(f"[ReActor V3] Using source face {source_face_index+1}/{len(source_faces)}, "
                  f"target face {target_face_index+1}/{len(target_faces)}")
            
            # Step 4: Perform face swap on FULL image first
            print(f"[ReActor V3] Swapping face...")
            swapped_img = self.swap_face(source_face, target_face, target_img.copy())
            
            # Step 5: Apply GPEN restoration if model is loaded
            # We restore the swapped result, not the original
            result = swapped_img
            if self.current_restorer is not None:
                print(f"[ReActor V3] Restoring with {self.current_restorer_name}...")
                
                # Now extract face from the SWAPPED image
                aligned_face, resolution = self.face_aligner.process_face(
                    swapped_img,  # Extract from swapped result
                    target_face,  # Use same face location
                    auto_resolution=auto_resolution
                )
                
                print(f"[ReActor V3] Face extraction - aligned_face: {aligned_face is not None}, resolution: {resolution}")
                
                if aligned_face is not None:
                    print(f"[ReActor V3] Running GPEN restoration at {resolution}x{resolution}...")
                    # Restore the swapped face with GPEN
                    restored_face = self.restore_face(aligned_face, resolution)
                    
                    print(f"[ReActor V3] Restoration complete - restored_face: {restored_face is not None}")
                    
                    if restored_face is not None:
                        print(f"[ReActor V3] Pasting restored face back...")
                        # Paste the restored face back into swapped image
                        result = self.face_aligner.paste_back(
                            swapped_img,  # Paste into swapped image, not original
                            restored_face,
                            target_face,
                            resolution
                        )
                        
                        print(f"[ReActor V3] Swapped and restored with {self.current_restorer_name} @ {resolution}x{resolution}")
                        status = (f"Success! Swapped and restored with {self.current_restorer_name} "
                                f"@ {resolution}x{resolution}")
                    else:
                        print(f"[ReActor V3] ERROR: Restoration returned None")
                        status = "Face swapped (restoration failed - returned None)"
                else:
                    print(f"[ReActor V3] ERROR: Could not extract/align face for restoration")
                    status = "Face swapped (could not align for restoration)"
            else:
                status = "Face swapped successfully (no restoration)"
            
            print(f"[ReActor V3] {status}")
            
            # Step 6: Automatic VRAM cleanup after processing
            if self.auto_cleanup:
                self.cleanup_memory(aggressive=False)
            
            return result, status
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            print(f"[ReActor V3] {error_msg}")
            traceback.print_exc()
            
            # Cleanup even on error
            if self.auto_cleanup:
                self.cleanup_memory(aggressive=False)
            
            return target_img, error_msg
    
    def get_available_restorers(self) -> List[str]:
        """Get list of available GPEN models"""
        return ['None'] + get_available_gpen_models(self.facerestore_path)
    
    def cleanup_memory(self, aggressive: bool = False):
        """
        Free VRAM and memory used by Reactor V3 models.
        
        This function ensures that after face swapping operations,
        VRAM is properly released for the next generation.
        
        Args:
            aggressive: If True, unload all cached models including GPEN
        """
        print(f"[ReActor V3] Cleaning up memory (aggressive={aggressive})...")
        
        try:
            # Step 1: Clear GPEN cache if aggressive cleanup
            if aggressive:
                if self.current_restorer is not None:
                    del self.current_restorer
                    self.current_restorer = None
                    self.current_restorer_name = None
                clear_gpen_cache()
            
            # Step 2: Use WebUI Forge's memory management if available
            if MEMORY_MANAGEMENT_AVAILABLE:
                memory_management.soft_empty_cache(force=True)
                print("[ReActor V3] Used Forge memory management for cleanup")
            else:
                # Fallback: Manual CUDA cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    print("[ReActor V3] Used manual CUDA cleanup")
            
            # Step 3: Python garbage collection
            gc.collect()
            
            # Step 4: Report memory status
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                print(f"[ReActor V3] VRAM - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
            
        except Exception as e:
            print(f"[ReActor V3] Warning during cleanup: {e}")


# Global instance (will be initialized by the UI script)
reactor_v3_engine: Optional[ReActorV3] = None


def get_reactor_v3_engine(models_path: str) -> ReActorV3:
    """
    Get or create the global ReActor V3 engine instance.
    
    Args:
        models_path: Path to models directory
        
    Returns:
        ReActorV3 engine instance
    """
    global reactor_v3_engine
    
    if reactor_v3_engine is None:
        reactor_v3_engine = ReActorV3(models_path)
    
    return reactor_v3_engine
