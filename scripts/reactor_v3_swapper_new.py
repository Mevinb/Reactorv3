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
    
    def initialize_face_analyser(self):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace not installed")
        
        if self.face_analyser is None:
            print("[ReActor V3] Initializing face analyzer...")
            self.face_analyser = FaceAnalysis(
                name='buffalo_l',
                root=self.insightface_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
            print("[ReActor V3] Face analyzer ready")
    
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
                gender_match: str = 'A') -> Tuple[np.ndarray, str]:
        """
        Simple workflow with gender matching:
        1. Swap face using InsightFace (with gender filtering)
        2. Restore entire image using GPEN+FaceRestoreHelper
        
        Args:
            gender_match: 'A' (all), 'M' (male only), 'F' (female only), 'S' (smart match)
        """
        try:
            # Initialize
            if self.face_analyser is None:
                self.initialize_face_analyser()
            if self.face_swapper is None:
                self.initialize_face_swapper()
            
            # Detect faces
            print("[ReActor V3] Detecting faces...")
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
            print("[ReActor V3] Swapping face...")
            result = self.face_swapper.get(target_img, target_face, source_face, paste_back=True)
            
            # Restore using GPEN if specified
            if restore_model:
                if not self.load_restorer(restore_model):
                    return result, "Swapped (restoration failed to load)"
                
                if self.current_restorer:
                    print(f"[ReActor V3] Restoring with {restore_model}...")
                    # This uses WebUI's FaceRestoreHelper - no custom masking!
                    restored_result = self.current_restorer.restore(result)
                    print(f"[ReActor V3] Swapped and restored with {restore_model}")
                    # Automatic VRAM cleanup after processing
                    if self.auto_cleanup:
                        self.cleanup_memory(aggressive=self.aggressive_cleanup)
                    return restored_result, f"Swapped and restored with {restore_model}"
            
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
