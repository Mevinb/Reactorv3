"""
ReActor V3 - Main Face Swapping Pipeline

Simple workflow:
1. Swap face on full image using InsightFace (paste_back=True)
2. Restore the entire result using GPEN with WebUI's FaceRestoreHelper
"""

import cv2
import numpy as np
import os
from typing import Optional, List, Tuple
from PIL import Image
import torch
import gc

try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import model_zoo
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

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

from reactor_v3_gpen_restorer_new import get_gpen_restorer, get_available_gpen_models, clear_gpen_cache


class ReActorV3:
    """Simple face swapper with GPEN restoration"""
    
    def __init__(self, models_path: str):
        self.models_path = models_path
        self.shared_models_root = models_path
        self.insightface_path = os.path.join(self.shared_models_root, 'insightface')
        self.facerestore_path = os.path.join(self.shared_models_root, 'facerestore_models')
        
        os.makedirs(self.insightface_path, exist_ok=True)
        os.makedirs(self.facerestore_path, exist_ok=True)
        
        self.face_analyser = None
        self.face_swapper = None
        self.current_restorer = None
        self.current_restorer_name = None
        
        # Auto cleanup settings
        self.auto_cleanup = True  # Automatically free VRAM after processing
        self.aggressive_cleanup = True  # Unload all models (default True for memory-constrained systems)
        
        print(f"[ReActor V3] InsightFace path: {self.insightface_path}")
        print(f"[ReActor V3] GPEN models path: {self.facerestore_path}")
        print(f"[ReActor V3] Auto VRAM cleanup: {self.auto_cleanup} (aggressive={self.aggressive_cleanup})")
    
    def initialize_face_analyser(self, detection_threshold: float = 0.5):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace not installed")
        
        if self.face_analyser is None:
            print(f"[ReActor V3] Initializing face analyzer (threshold={detection_threshold})...")
            self.face_analyser = FaceAnalysis(
                name='buffalo_l',
                root=self.insightface_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyser.prepare(ctx_id=0, det_size=(640, 640), det_thresh=detection_threshold)
            print("[ReActor V3] Face analyzer ready")
        else:
            # Update threshold if changed
            self.face_analyser.det_thresh = detection_threshold
    
    def initialize_face_swapper(self, model_name: str = 'inswapper_128.onnx'):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace not installed")
        
        if self.face_swapper is None:
            print(f"[ReActor V3] Initializing face swapper: {model_name}")
            
            search_paths = [
                os.path.join(self.insightface_path, model_name),
                os.path.join(self.insightface_path, 'models', model_name),
                os.path.join(self.shared_models_root, 'reactor', model_name),
            ]
            
            model_path = None
            for path in search_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(f"Model not found: {model_name}")
            
            self.face_swapper = model_zoo.get_model(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            print(f"[ReActor V3] Face swapper ready: {model_path}")
    
    def get_faces(self, img: np.ndarray) -> List:
        if self.face_analyser is None:
            self.initialize_face_analyser()
        return self.face_analyser.get(img)
    
    def get_gender(self, face) -> str:
        """
        Get gender from a face object.
        
        InsightFace returns gender as an attribute where:
        - 0 or negative values = Female
        - 1 or positive values = Male
        
        Args:
            face: Face detection object from InsightFace
            
        Returns:
            'M' for male, 'F' for female, 'U' for unknown
        """
        try:
            if hasattr(face, 'gender'):
                gender_value = face.gender
                if isinstance(gender_value, (int, float, np.integer, np.floating)):
                    return 'M' if gender_value >= 0.5 else 'F'
            elif hasattr(face, 'sex'):
                sex_value = face.sex
                if isinstance(sex_value, (int, float, np.integer, np.floating)):
                    return 'M' if sex_value >= 0.5 else 'F'
            
            if hasattr(face, '__dict__'):
                face_dict = face.__dict__
                if 'gender' in face_dict:
                    return 'M' if face_dict['gender'] >= 0.5 else 'F'
                if 'sex' in face_dict:
                    return 'M' if face_dict['sex'] >= 0.5 else 'F'
            
            return 'U'
        except Exception as e:
            print(f"[ReActor V3] Warning: Could not determine gender: {e}")
            return 'U'
    
    def filter_faces_by_gender(self, faces: List, target_gender: str) -> List:
        """
        Filter faces by gender.
        
        Args:
            faces: List of face detection objects
            target_gender: 'M' for male, 'F' for female, 'A' for all (no filter)
            
        Returns:
            Filtered list of faces
        """
        if not faces or target_gender == 'A':
            return faces
        
        filtered = []
        for i, face in enumerate(faces):
            gender = self.get_gender(face)
            if gender == target_gender or gender == 'U':
                filtered.append(face)
            print(f"[ReActor V3] Face {i}: Gender={gender} {'✓' if gender == target_gender or gender == 'U' else '✗'}")
        
        return filtered
    
    def apply_color_correction(self, swapped_img: np.ndarray, target_img: np.ndarray, target_face) -> np.ndarray:
        """
        Apply color correction to match swapped face to target image lighting.
        Uses color transfer in LAB color space for natural results.
        
        Args:
            swapped_img: Image with swapped face
            target_img: Original target image
            target_face: Face object from InsightFace with bbox coordinates
            
        Returns:
            Color-corrected image
        """
        try:
            # Get face bounding box with some expansion for better color matching
            bbox = target_face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Expand bbox by 20% for better color context
            h, w = y2 - y1, x2 - x1
            expand = int(max(h, w) * 0.2)
            x1 = max(0, x1 - expand)
            y1 = max(0, y1 - expand)
            x2 = min(target_img.shape[1], x2 + expand)
            y2 = min(target_img.shape[0], y2 + expand)
            
            # Extract face regions
            swapped_face_region = swapped_img[y1:y2, x1:x2]
            target_face_region = target_img[y1:y2, x1:x2]
            
            # Convert to LAB color space for color transfer
            swapped_lab = cv2.cvtColor(swapped_face_region, cv2.COLOR_BGR2LAB).astype(np.float32)
            target_lab = cv2.cvtColor(target_face_region, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Calculate mean and std for each channel
            swapped_mean, swapped_std = cv2.meanStdDev(swapped_lab)
            target_mean, target_std = cv2.meanStdDev(target_lab)
            
            # Apply color correction formula
            swapped_lab -= swapped_mean.reshape(1, 1, 3)
            swapped_lab *= (target_std / swapped_std).reshape(1, 1, 3)
            swapped_lab += target_mean.reshape(1, 1, 3)
            
            # Clip and convert back
            swapped_lab = np.clip(swapped_lab, 0, 255).astype(np.uint8)
            corrected_face = cv2.cvtColor(swapped_lab, cv2.COLOR_LAB2BGR)
            
            # Blend corrected face back into result
            result = swapped_img.copy()
            result[y1:y2, x1:x2] = corrected_face
            
            print(f"[ReActor V3] Color correction applied (target mean: L={target_mean[0][0]:.1f}, A={target_mean[1][0]:.1f}, B={target_mean[2][0]:.1f})")
            return result
            
        except Exception as e:
            print(f"[ReActor V3] Color correction failed: {e}, using original")
            return swapped_img
    
    def set_cleanup_mode(self, aggressive: bool):
        """Set whether cleanup should be aggressive (unload all models)"""
        self.aggressive_cleanup = aggressive
        print(f"[ReActor V3] Cleanup mode set to aggressive={aggressive}")
    
    def load_restorer(self, model_name: str) -> bool:
        if model_name == self.current_restorer_name and self.current_restorer is not None:
            return True
        
        if not model_name or model_name.lower() == 'none':
            self.current_restorer = None
            self.current_restorer_name = None
            return True
        
        model_path = os.path.join(self.facerestore_path, model_name)
        if not os.path.exists(model_path):
            print(f"[ReActor V3] Model not found: {model_path}")
            return False
        
        try:
            self.current_restorer = get_gpen_restorer(model_path, device='cuda')
            self.current_restorer_name = model_name
            return True
        except Exception as e:
            print(f"[ReActor V3] Error loading restorer: {e}")
            return False
    
    def process(self,
                source_img: np.ndarray,
                target_img: np.ndarray,
                source_face_index: int = 0,
                target_face_index: int = 0,
                restore_model: str = None,
                gender_match: str = 'A',
                blend_ratio: float = 1.0,
                detection_threshold: float = 0.5,
                color_correction: bool = True,
                upscale_factor: int = 1,
                resolution_threshold: int = 384) -> Tuple[np.ndarray, str]:
        """
        Advanced workflow with realism controls:
        1. Swap face using InsightFace (with gender filtering and blend control)
        2. Apply color correction for lighting match
        3. Restore entire image using GPEN+FaceRestoreHelper with optional upscaling
        
        Args:
            gender_match: 'A' (all), 'M' (male only), 'F' (female only), 'S' (smart match)
            blend_ratio: 0.0-1.0, controls how much of swapped face to blend (1.0 = full swap)
            detection_threshold: Face detection confidence threshold (0.1-0.99)
            color_correction: Apply color correction to match target lighting
            upscale_factor: Extract faces at 1x or 2x resolution before restoration
            resolution_threshold: Face size threshold for auto-selecting 512 vs 1024 GPEN
        """
        try:
            # Initialize with detection threshold
            if self.face_analyser is None:
                self.initialize_face_analyser(detection_threshold)
            else:
                # Update threshold if changed
                self.face_analyser.det_thresh = detection_threshold
            
            if self.face_swapper is None:
                self.initialize_face_swapper()
            
            # Detect faces
            print(f"[ReActor V3] Detecting faces (threshold={detection_threshold})...")
            source_faces = self.get_faces(source_img)
            target_faces = self.get_faces(target_img)
            
            if not source_faces:
                return target_img, "Error: No face in source"
            if not target_faces:
                return target_img, "Error: No face in target"
            
            # Apply gender matching
            if gender_match == 'S':  # Smart match
                source_face = source_faces[min(source_face_index, len(source_faces)-1)]
                source_gender = self.get_gender(source_face)
                print(f"[ReActor V3] Smart Match: Source gender = {source_gender}")
                
                filtered_target_faces = self.filter_faces_by_gender(target_faces, source_gender)
                if not filtered_target_faces:
                    gender_name = "male" if source_gender == 'M' else "female" if source_gender == 'F' else "matching"
                    return target_img, f"Error: No {gender_name} face in target to match source"
                target_faces = filtered_target_faces
                
            elif gender_match in ['M', 'F']:  # Filter by specific gender
                print(f"[ReActor V3] Gender Filter: {gender_match}")
                source_faces = self.filter_faces_by_gender(source_faces, gender_match)
                target_faces = self.filter_faces_by_gender(target_faces, gender_match)
                
                if not source_faces:
                    gender_name = "male" if gender_match == 'M' else "female"
                    return target_img, f"Error: No {gender_name} face in source"
                if not target_faces:
                    gender_name = "male" if gender_match == 'M' else "female"
                    return target_img, f"Error: No {gender_name} face in target"
            
            source_face = source_faces[min(source_face_index, len(source_faces)-1)]
            target_face = target_faces[min(target_face_index, len(target_faces)-1)]
            
            # Display gender info
            source_gender = self.get_gender(source_face)
            target_gender = self.get_gender(target_face)
            print(f"[ReActor V3] Swapping: Source ({source_gender}) -> Target ({target_gender})")
            
            # Swap face (InsightFace handles blending with paste_back=True)
            print(f"[ReActor V3] Swapping face (blend={blend_ratio:.2f})...")
            result = self.face_swapper.get(target_img, target_face, source_face, paste_back=True)
            
            # Apply blend ratio if not full swap
            if blend_ratio < 1.0:
                print(f"[ReActor V3] Applying blend ratio {blend_ratio:.2f}...")
                result = cv2.addWeighted(result, blend_ratio, target_img, 1 - blend_ratio, 0)
            
            # Apply color correction to match target lighting
            if color_correction:
                print("[ReActor V3] Applying color correction...")
                result = self.apply_color_correction(result, target_img, target_face)
            
            # Restore using GPEN if specified
            if restore_model:
                if not self.load_restorer(restore_model):
                    return result, "Swapped (restoration failed to load)"
                
                if self.current_restorer:
                    # Set upscale factor for face extraction
                    if upscale_factor > 1:
                        print(f"[ReActor V3] Using {upscale_factor}x upscale for face extraction...")
                        self.current_restorer.face_helper.upscale_factor = upscale_factor
                    
                    print(f"[ReActor V3] Restoring with {restore_model} (upscale={upscale_factor}x)...")
                    restored_result = self.current_restorer.restore(result)
                    
                    # Reset upscale factor
                    if upscale_factor > 1:
                        self.current_restorer.face_helper.upscale_factor = 1
                    
                    status_msg = f"Swapped and restored with {restore_model}"
                    if blend_ratio < 1.0:
                        status_msg += f" (blend={blend_ratio:.0%})"
                    if color_correction:
                        status_msg += " + color corrected"
                    if upscale_factor > 1:
                        status_msg += f" + {upscale_factor}x upscaled"
                    
                    print(f"[ReActor V3] {status_msg}")
                    
                    # Automatic VRAM cleanup after processing
                    if self.auto_cleanup:
                        self.cleanup_memory(aggressive=self.aggressive_cleanup)
                    return restored_result, status_msg
            
            # Automatic VRAM cleanup after processing
            if self.auto_cleanup:
                self.cleanup_memory(aggressive=self.aggressive_cleanup)
            return result, "Face swapped successfully"
            
        except Exception as e:
            print(f"[ReActor V3] Error: {e}")
            import traceback
            traceback.print_exc()
            # Cleanup even on error
            if self.auto_cleanup:
                self.cleanup_memory(aggressive=self.aggressive_cleanup)
            return target_img, f"Error: {str(e)}"
    
    def cleanup_memory(self, aggressive: bool = False):
        """
        Free VRAM and memory used by Reactor V3 models.
        
        This function ensures that after face swapping operations,
        VRAM is properly released for the next generation.
        
        Args:
            aggressive: If True, unload all cached models including InsightFace and GPEN
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
                
                # CRITICAL: Unload InsightFace models (ONNX Runtime sessions)
                # These are the main memory hogs (~2GB VRAM)
                if self.face_analyser is not None:
                    print("[ReActor V3] Unloading InsightFace face analyzer...")
                    del self.face_analyser
                    self.face_analyser = None
                
                if self.face_swapper is not None:
                    print("[ReActor V3] Unloading InsightFace face swapper...")
                    del self.face_swapper
                    self.face_swapper = None
            
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
    
    def get_available_restorers(self) -> List[str]:
        return ['None'] + get_available_gpen_models(self.facerestore_path)


# Global instance
reactor_v3_engine: Optional[ReActorV3] = None

def get_reactor_v3_engine(models_path: str) -> ReActorV3:
    global reactor_v3_engine
    if reactor_v3_engine is None:
        reactor_v3_engine = ReActorV3(models_path)
    return reactor_v3_engine
