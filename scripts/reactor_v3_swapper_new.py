"""
ReActor V3 - Main Face Swapping Pipeline

Simple workflow:
1. Swap face on full image using InsightFace (paste_back=True)
2. Restore the entire result using GPEN with WebUI's FaceRestoreHelper
"""

import cv2
import numpy as np
import os
import time
from typing import Optional, List, Tuple
from PIL import Image
import torch
import gc

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
from reactor_v3_face_fixer import (
    auto_fix_face,
    compute_identity_restore_weight,
    compute_resolution_restore_limit,
)


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
        self.auto_face_fix = True  # Automatically fix face sharpness/texture to match reference
        self.occlusion_enabled = True
        self.occlusion_strength = 1.0
        self.occlusion_sensitivity = 0.55
        
        # Mouth preservation settings
        self.mouth_protect_enabled = True
        self.mouth_protect_strength = 0.75  # How strongly to blend original mouth back (0-1)
        self.mouth_open_threshold = 0.28     # Mouth-open ratio above which preservation kicks in
        
        print(f"[ReActor V3] InsightFace path: {self.insightface_path}")
        print(f"[ReActor V3] GPEN models path: {self.facerestore_path}")
        print(f"[ReActor V3] Auto VRAM cleanup: {self.auto_cleanup} (aggressive={self.aggressive_cleanup})")
        print(f"[ReActor V3] Auto face fix (match reference detail): {self.auto_face_fix}")
        print(
            f"[ReActor V3] Adaptive occlusion: enabled={self.occlusion_enabled}, "
            f"strength={self.occlusion_strength:.2f}, sensitivity={self.occlusion_sensitivity:.2f}"
        )
        print(
            f"[ReActor V3] Mouth protection: enabled={self.mouth_protect_enabled}, "
            f"strength={self.mouth_protect_strength:.2f}, threshold={self.mouth_open_threshold:.2f}"
        )
    
    def initialize_face_analyser(self):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace not installed")
        
        if self.face_analyser is None:
            print("[ReActor V3] ── Initializing Face Analyzer ──")
            print(f"[ReActor V3]   Model: buffalo_l")
            print(f"[ReActor V3]   Root: {self.insightface_path}")
            print(f"[ReActor V3]   Providers: CUDAExecutionProvider, CPUExecutionProvider")
            t0 = time.time()
            self.face_analyser = FaceAnalysis(
                name='buffalo_l',
                root=self.insightface_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
            elapsed = time.time() - t0
            print(f"[ReActor V3]   Face analyzer ready (loaded in {elapsed:.2f}s)")
            if torch.cuda.is_available():
                vram_alloc = torch.cuda.memory_allocated() / (1024**3)
                vram_res = torch.cuda.memory_reserved() / (1024**3)
                print(f"[ReActor V3]   VRAM after analyzer load - Allocated: {vram_alloc:.2f} GB, Reserved: {vram_res:.2f} GB")
    
    def initialize_face_swapper(self, model_name: str = 'inswapper_128.onnx'):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace not installed")
        
        if self.face_swapper is None:
            print(f"[ReActor V3] ── Initializing Face Swapper ──")
            print(f"[ReActor V3]   Model name: {model_name}")
            
            search_paths = [
                os.path.join(self.insightface_path, model_name),
                os.path.join(self.insightface_path, 'models', model_name),
                os.path.join(self.shared_models_root, 'reactor', model_name),
            ]
            
            model_path = None
            for path in search_paths:
                print(f"[ReActor V3]   Searching: {path} ... {'FOUND' if os.path.exists(path) else 'not found'}")
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(f"Model not found: {model_name}")
            
            t0 = time.time()
            self.face_swapper = model_zoo.get_model(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            elapsed = time.time() - t0
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"[ReActor V3]   Face swapper ready: {model_path}")
            print(f"[ReActor V3]   Model size: {model_size_mb:.1f} MB, loaded in {elapsed:.2f}s")
            if torch.cuda.is_available():
                vram_alloc = torch.cuda.memory_allocated() / (1024**3)
                vram_res = torch.cuda.memory_reserved() / (1024**3)
                print(f"[ReActor V3]   VRAM after swapper load - Allocated: {vram_alloc:.2f} GB, Reserved: {vram_res:.2f} GB")
    
    def get_faces(self, img: np.ndarray) -> List:
        if self.face_analyser is None:
            self.initialize_face_analyser()
        t0 = time.time()
        faces = self.face_analyser.get(img)
        elapsed = time.time() - t0
        h, w = img.shape[:2]
        print(f"[ReActor V3]   Face detection on {w}x{h} image: found {len(faces)} face(s) in {elapsed:.3f}s")
        for i, face in enumerate(faces):
            bbox = [int(v) for v in face.bbox] if hasattr(face, 'bbox') else 'N/A'
            det_score = f"{face.det_score:.3f}" if hasattr(face, 'det_score') else 'N/A'
            age = getattr(face, 'age', 'N/A')
            gender = self.get_gender(face)
            print(f"[ReActor V3]     Face {i}: bbox={bbox}, det_score={det_score}, gender={gender}, age={age}")
        return faces
    
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
    
    # ── Auto Face Match Methods ──────────────────────────────────────────────

    @staticmethod
    def compute_face_similarity(face1, face2) -> float:
        """
        Compute cosine similarity between two face embeddings (ArcFace 512-dim).
        
        InsightFace provides 512-dimensional ArcFace embeddings for each detected face.
        Cosine similarity between these embeddings reliably measures face identity:
          > 0.6  : Very likely same person
          0.3-0.6: Possibly same person (similar features, same gender/ethnicity)
          < 0.3  : Different people
        
        Returns:
            float: cosine similarity in range [-1, 1], higher = more similar
        """
        if not hasattr(face1, 'embedding') or not hasattr(face2, 'embedding'):
            return 0.0
        emb1 = face1.embedding
        emb2 = face2.embedding
        if emb1 is None or emb2 is None:
            return 0.0
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        emb1_n = emb1 / norm1
        emb2_n = emb2 / norm2
        return float(np.dot(emb1_n, emb2_n))

    def find_best_matches(self, source_faces: List, target_faces: List,
                          threshold: float = 0.3, gender_match: str = 'A') -> List[Tuple[int, int, float]]:
        """
        Find optimal 1:1 source→target face matches using embedding cosine similarity.
        Uses Hungarian algorithm (scipy) for globally optimal assignment,
        with greedy fallback if scipy is unavailable.
        
        Args:
            source_faces: List of InsightFace face objects from source image(s)
            target_faces: List of InsightFace face objects from target image
            threshold: Minimum cosine similarity to accept a match (0-1)
            gender_match: 'A' = all, 'S' = only match same gender, 'M'/'F' = filter
            
        Returns:
            List of (source_idx, target_idx, similarity) tuples, sorted by similarity desc
        """
        if not source_faces or not target_faces:
            return []
        
        n_src = len(source_faces)
        n_tgt = len(target_faces)
        
        # Build pairwise similarity matrix using ArcFace embeddings
        sim_matrix = np.zeros((n_src, n_tgt), dtype=np.float32)
        for i, sf in enumerate(source_faces):
            for j, tf in enumerate(target_faces):
                sim = self.compute_face_similarity(sf, tf)
                
                # Gender penalty: reduce similarity if genders don't match (Smart mode)
                if gender_match == 'S':
                    sg = self.get_gender(sf)
                    tg = self.get_gender(tf)
                    if sg != 'U' and tg != 'U' and sg != tg:
                        sim *= 0.1  # Heavy penalty for gender mismatch
                
                sim_matrix[i, j] = sim
        
        # Log the similarity matrix
        print(f"[ReActor V3] ── Face Similarity Matrix ({n_src} source × {n_tgt} target) ──")
        for i in range(n_src):
            sg = self.get_gender(source_faces[i])
            row_str = ", ".join([f"T{j}={sim_matrix[i,j]:.3f}" for j in range(n_tgt)])
            print(f"[ReActor V3]   S{i}({sg}): [{row_str}]")
        
        # Try Hungarian algorithm (globally optimal) first
        try:
            from scipy.optimize import linear_sum_assignment
            max_dim = max(n_src, n_tgt)
            # Pad to square matrix (Hungarian requires square)
            padded_cost = np.zeros((max_dim, max_dim), dtype=np.float64)
            padded_cost[:n_src, :n_tgt] = -sim_matrix.astype(np.float64)  # negate for minimization
            row_ind, col_ind = linear_sum_assignment(padded_cost)
            
            matches = []
            for r, c in zip(row_ind, col_ind):
                if r < n_src and c < n_tgt:
                    sim = float(sim_matrix[r, c])
                    if sim >= threshold:
                        matches.append((int(r), int(c), sim))
            matches.sort(key=lambda x: -x[2])
            print(f"[ReActor V3]   Matching algorithm: Hungarian (globally optimal)")
        except ImportError:
            # Greedy matching fallback - works well for small face counts
            matches = []
            used_sources = set()
            used_targets = set()
            
            # Flatten and sort all pairs by similarity (descending)
            pairs = []
            for i in range(n_src):
                for j in range(n_tgt):
                    pairs.append((i, j, float(sim_matrix[i, j])))
            pairs.sort(key=lambda x: -x[2])
            
            for src_idx, tgt_idx, sim in pairs:
                if src_idx in used_sources or tgt_idx in used_targets:
                    continue
                if sim < threshold:
                    break
                matches.append((src_idx, tgt_idx, sim))
                used_sources.add(src_idx)
                used_targets.add(tgt_idx)
            
            print(f"[ReActor V3]   Matching algorithm: Greedy")
        
        # Log matches
        if matches:
            for src_idx, tgt_idx, sim in matches:
                sg = self.get_gender(source_faces[src_idx])
                tg = self.get_gender(target_faces[tgt_idx])
                print(f"[ReActor V3]   ✓ Match: Source[{src_idx}]({sg}) → Target[{tgt_idx}]({tg}) [cosine={sim:.3f}]")
        else:
            print(f"[ReActor V3]   ✗ No matches found above threshold {threshold}")
        
        return matches

    def process_auto_match(self,
                           source_img: np.ndarray,
                           target_img: np.ndarray,
                           restore_model: str = None,
                           gender_match: str = 'A',
                           similarity_threshold: float = 0.3,
                           additional_sources: List[np.ndarray] = None) -> Tuple[np.ndarray, str]:
        """
        Automatic multi-face detection, matching, and swapping pipeline.
        
        This is the core auto-detection feature:
        1. Detect ALL faces in source image(s) and target image
        2. Compute ArcFace embedding cosine similarity for every source↔target pair
        3. Find optimal 1:1 assignment (Hungarian algorithm / greedy)
        4. Swap each matched pair sequentially
        5. Apply GPEN restoration + harmonization on the final result
        
        Handles any combination:
        - 1 source face, 1 target face: Simple swap (like original)
        - 1 source face, N target faces: Swaps only the best-matching target face
        - N source faces, M target faces: Optimally matches and swaps all pairs
        
        Args:
            source_img: Primary source image (BGR numpy, can contain multiple faces)
            target_img: Target image to swap faces in (BGR numpy)
            restore_model: GPEN model name (None or 'None' to skip restoration)
            gender_match: 'A' (all), 'S' (smart/same gender only), 'M'/'F' (filter)
            similarity_threshold: Minimum cosine similarity to accept a match (0.0-1.0)
            additional_sources: Optional list of extra source images (BGR numpy arrays)
            
        Returns:
            (result_image, status_message)
        """
        try:
            print("")
            print("[ReActor V3] ════════════════════════════════════════")
            print("[ReActor V3]   AUTO FACE MATCH PIPELINE START")
            print("[ReActor V3] ════════════════════════════════════════")
            print(f"[ReActor V3]   Source image: {source_img.shape[1]}x{source_img.shape[0]}")
            print(f"[ReActor V3]   Target image: {target_img.shape[1]}x{target_img.shape[0]}")
            print(f"[ReActor V3]   Gender match: {gender_match}")
            print(f"[ReActor V3]   Similarity threshold: {similarity_threshold}")
            print(f"[ReActor V3]   Restore model: {restore_model or 'None'}")
            if additional_sources:
                print(f"[ReActor V3]   Additional source images: {len(additional_sources)}")
            pipeline_start = time.time()
            
            # Initialize models
            if self.face_analyser is None:
                self.initialize_face_analyser()
            if self.face_swapper is None:
                self.initialize_face_swapper()
            
            # ── Detect faces in all images ──
            print("[ReActor V3] ── Detecting All Faces ──")
            print("[ReActor V3]   Scanning primary source...")
            source_faces = self.get_faces(source_img)
            all_source_faces = list(source_faces) if source_faces else []
            
            # Add faces from additional source images
            if additional_sources:
                for i, add_src in enumerate(additional_sources):
                    if add_src is not None and add_src.size > 0:
                        print(f"[ReActor V3]   Scanning additional source {i+1}...")
                        add_faces = self.get_faces(add_src)
                        if add_faces:
                            print(f"[ReActor V3]     Found {len(add_faces)} face(s) in additional source {i+1}")
                            all_source_faces.extend(add_faces)
            
            print("[ReActor V3]   Scanning target image...")
            target_faces = self.get_faces(target_img)
            
            if not all_source_faces:
                msg = "Error: No faces detected in source image(s)"
                print(f"[ReActor V3]   {msg}")
                return target_img, msg
            if not target_faces:
                msg = "Error: No faces detected in target image"
                print(f"[ReActor V3]   {msg}")
                return target_img, msg
            
            print(f"[ReActor V3]   Total source faces: {len(all_source_faces)}")
            print(f"[ReActor V3]   Total target faces: {len(target_faces)}")
            
            # ── Apply gender filter if M/F ──
            if gender_match in ['M', 'F']:
                all_source_faces = self.filter_faces_by_gender(all_source_faces, gender_match)
                target_faces = self.filter_faces_by_gender(target_faces, gender_match)
                if not all_source_faces:
                    gender_name = "male" if gender_match == 'M' else "female"
                    return target_img, f"Error: No {gender_name} faces in source"
                if not target_faces:
                    gender_name = "male" if gender_match == 'M' else "female"
                    return target_img, f"Error: No {gender_name} faces in target"
            
            # ── Find optimal face matches ──
            print("[ReActor V3] ── Computing Optimal Face Matches ──")
            matches = self.find_best_matches(
                all_source_faces, target_faces,
                threshold=similarity_threshold,
                gender_match=gender_match
            )
            
            if not matches:
                # Fallback: if no matches above threshold, use the single best match
                # This ensures at least one swap happens even with low similarity
                print("[ReActor V3]   No matches above threshold — using best available pair as fallback")
                best_sim = -1.0
                best_pair = None
                for i, sf in enumerate(all_source_faces):
                    for j, tf in enumerate(target_faces):
                        sim = self.compute_face_similarity(sf, tf)
                        if sim > best_sim:
                            best_sim = sim
                            best_pair = (i, j, sim)
                if best_pair:
                    matches = [best_pair]
                    sg = self.get_gender(all_source_faces[best_pair[0]])
                    tg = self.get_gender(target_faces[best_pair[1]])
                    print(f"[ReActor V3]   Fallback: S{best_pair[0]}({sg}) → T{best_pair[1]}({tg}) [sim={best_pair[2]:.3f}]")
                else:
                    return target_img, "Error: Could not match any faces"
            
            print(f"[ReActor V3]   Total matches: {len(matches)}")
            
            # ── Swap each matched pair ──
            result = target_img.copy()
            original_for_occlusion = target_img.copy()
            
            # Sort by target face x-position (right-to-left) to minimize bbox shift
            def _sort_key(match):
                tf = target_faces[match[1]]
                return -float(tf.bbox[0]) if hasattr(tf, 'bbox') else 0
            matches.sort(key=_sort_key)
            
            print("[ReActor V3] ── Swapping Matched Faces ──")
            swapped_target_faces = []  # Track which target faces were swapped
            for mi, (src_idx, tgt_idx, similarity) in enumerate(matches):
                source_face = all_source_faces[src_idx]
                target_face = target_faces[tgt_idx]
                sg = self.get_gender(source_face)
                tg = self.get_gender(target_face)
                
                print(f"[ReActor V3]   Swap {mi+1}/{len(matches)}: Source[{src_idx}]({sg}) → Target[{tgt_idx}]({tg}) [sim={similarity:.3f}]")
                
                if hasattr(target_face, 'bbox'):
                    bbox = [int(v) for v in target_face.bbox]
                    print(f"[ReActor V3]     Target face bbox: {bbox}")
                if hasattr(source_face, 'embedding') and source_face.embedding is not None:
                    emb_norm = float(np.linalg.norm(source_face.embedding))
                    print(f"[ReActor V3]     Source embedding norm: {emb_norm:.2f}")
                
                t0 = time.time()
                result = self.face_swapper.get(result, target_face, source_face, paste_back=True)
                elapsed = time.time() - t0
                print(f"[ReActor V3]     Swap completed in {elapsed:.3f}s")
                
                # Mouth preservation per face
                if self.mouth_protect_enabled:
                    mouth_ratio, mouth_is_open = self._detect_mouth_open(target_face, original_for_occlusion)
                    if mouth_is_open:
                        print(f"[ReActor V3]     ⚠ Open mouth on Target[{tgt_idx}] (ratio={mouth_ratio:.3f}) — preserving")
                        result = self._preserve_mouth_region(original_for_occlusion, result, target_face, mouth_ratio)
                
                # Occlusion preservation per face
                result = self._preserve_foreground_occlusions(
                    original_for_occlusion, result, target_face, stage=f"auto-swap-{mi}"
                )
                swapped_target_faces.append(target_face)
            
            # ── Restore with GPEN if specified ──
            if restore_model and restore_model.lower() != 'none':
                if self.load_restorer(restore_model):
                    if self.current_restorer:
                        print(f"[ReActor V3] ── Restoring with {restore_model} ──")
                        t0 = time.time()
                        restored = self.current_restorer.restore(result)
                        elapsed = time.time() - t0
                        print(f"[ReActor V3]   Restoration completed in {elapsed:.3f}s")
                        
                        # Identity-aware + resolution-aware harmonization per face (Sections B, F)
                        print("[ReActor V3] ── Identity-Aware Harmonization (Auto-Match) ──")
                        swapped_emb = self._get_face_embedding(result)
                        restored_emb = self._get_face_embedding(restored)

                        for mi_h, tf in enumerate(swapped_target_faces):
                            src_face_h = all_source_faces[matches[mi_h][0]] if mi_h < len(matches) else None
                            src_emb_h = src_face_h.embedding if (src_face_h and hasattr(src_face_h, 'embedding')) else None

                            id_weight = compute_identity_restore_weight(
                                src_emb_h, swapped_emb, restored_emb, base_restore_weight=0.8
                            )
                            res_limit = compute_resolution_restore_limit(
                                tf.bbox if hasattr(tf, 'bbox') else None
                            )
                            eff_weight = min(id_weight, res_limit)
                            restored = self._harmonize_restored_face(
                                result, restored, tf, override_restore_weight=eff_weight,
                            )
                        
                        # Final occlusion preservation for each face
                        for mi, tf in enumerate(swapped_target_faces):
                            restored = self._preserve_foreground_occlusions(
                                original_for_occlusion, restored, tf, stage=f"auto-restore-{mi}"
                            )
                        
                        result = restored
            
            # ── Adaptive Face Enhancement (replaces static face fix) ──
            if self.auto_face_fix:
                print("[ReActor V3] ── Adaptive Face Enhancement (Auto-Match) ──")
                t0_fix = time.time()
                # Use the first source face for identity anchor
                primary_source_face = all_source_faces[0] if all_source_faces else None
                result = auto_fix_face(
                    reference_img=source_img,
                    output_img=result,
                    face_analyser=self.face_analyser,
                    target_img=target_img,
                    source_face=primary_source_face,
                )
                fix_elapsed = time.time() - t0_fix
                print(f"[ReActor V3]   Adaptive face enhancement completed in {fix_elapsed:.3f}s")

            # ── Cleanup ──
            if self.auto_cleanup:
                self.cleanup_memory(aggressive=self.aggressive_cleanup)
            
            pipeline_elapsed = time.time() - pipeline_start
            status = f"Auto-matched and swapped {len(matches)} face(s) in {pipeline_elapsed:.2f}s"
            print(f"[ReActor V3] ✓ {status}")
            print("[ReActor V3] ════════════════════════════════════════")
            print("")
            return result, status
        
        except Exception as e:
            print(f"[ReActor V3] ✗ Auto-match ERROR: {e}")
            import traceback
            traceback.print_exc()
            if self.auto_cleanup:
                self.cleanup_memory(aggressive=self.aggressive_cleanup)
            return target_img, f"Error: {str(e)}"

    # ── End Auto Face Match Methods ──────────────────────────────────────────

    def set_mouth_protection(self, enabled: bool, strength: float = 0.75, threshold: float = 0.28):
        """Configure mouth-region preservation for open-mouth targets."""
        self.mouth_protect_enabled = bool(enabled)
        self.mouth_protect_strength = float(np.clip(strength, 0.0, 1.0))
        self.mouth_open_threshold = float(np.clip(threshold, 0.05, 0.70))
        print(
            f"[ReActor V3] Mouth protection set: enabled={self.mouth_protect_enabled}, "
            f"strength={self.mouth_protect_strength:.2f}, threshold={self.mouth_open_threshold:.2f}"
        )

    # ── Mouth Detection & Preservation ────────────────────────────────────────

    def _detect_mouth_open(self, face, img: np.ndarray) -> Tuple[float, bool]:
        """
        Detect whether a face has its mouth open using multi-signal analysis.

        InsightFace buffalo_l provides 5 key-points (kps):
          [0] left_eye, [1] right_eye, [2] nose, [3] mouth_left, [4] mouth_right

        Additionally, if the model provides 68/106/478 landmarks we use the
        actual lip landmarks for much better accuracy.

        Strategy (5-kps fallback):
          - Estimate mouth center from mean of mouth_left + mouth_right
          - Measure vertical gap between nose tip and mouth center
          - Normalise by inter-eye distance (robust to face scale / distance)
          - Cross-check with a local texture analysis: open mouths have a
            dark cavity between the lips → high intensity variance in a
            small strip below the midpoint of the two mouth corners.

        Returns:
            (mouth_open_ratio, is_open)  where ratio is 0-1 normalised.
        """
        threshold = self.mouth_open_threshold

        # ── Try detailed landmarks first (68-pt / 106-pt / 478-pt) ────────
        landmark_2d = getattr(face, 'landmark_2d_106', None)
        if landmark_2d is None:
            landmark_2d = getattr(face, 'landmark_3d_68', None)
        if landmark_2d is None:
            landmark_2d = getattr(face, 'landmark_2d_68', None)

        if landmark_2d is not None:
            lm = np.asarray(landmark_2d, dtype=np.float32)
            if lm.ndim == 2:
                if lm.shape[0] >= 106:
                    # 106-point model: upper lip 76, lower lip 86 (approx)
                    upper_lip = lm[82]  # centre-top of upper lip
                    lower_lip = lm[87]  # centre-bottom of lower lip
                    left_eye = lm[35]
                    right_eye = lm[93]
                elif lm.shape[0] >= 68:
                    # 68-point model: inner lip points 61-67
                    upper_lip = lm[62]  # top of inner upper lip
                    lower_lip = lm[66]  # bottom of inner lower lip
                    left_eye = lm[36]
                    right_eye = lm[45]
                else:
                    upper_lip = lower_lip = left_eye = right_eye = None

                if upper_lip is not None:
                    lip_gap = float(np.linalg.norm(lower_lip[:2] - upper_lip[:2]))
                    eye_dist = float(np.linalg.norm(right_eye[:2] - left_eye[:2]))
                    if eye_dist < 1.0:
                        eye_dist = 1.0
                    ratio = lip_gap / eye_dist
                    is_open = ratio > threshold
                    print(
                        f"[ReActor V3]   Mouth (landmark): lip_gap={lip_gap:.1f}px, "
                        f"eye_dist={eye_dist:.1f}px, ratio={ratio:.3f}, "
                        f"open={'YES' if is_open else 'no'} (thr={threshold:.2f})"
                    )
                    return ratio, is_open

        # ── Fallback: 5 key-points + texture analysis ─────────────────────
        kps = getattr(face, 'kps', None)
        if kps is None or len(kps) < 5:
            return 0.0, False

        kps = np.asarray(kps, dtype=np.float32)
        left_eye, right_eye = kps[0], kps[1]
        nose = kps[2]
        mouth_l, mouth_r = kps[3], kps[4]

        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        if eye_dist < 1.0:
            eye_dist = 1.0

        mouth_center = (mouth_l + mouth_r) / 2.0
        nose_to_mouth = float(np.linalg.norm(mouth_center - nose))

        # Geometric ratio: nose-to-mouth / inter-eye distance
        geom_ratio = nose_to_mouth / eye_dist

        # Texture check: sample a small patch below the mouth centre
        # Open mouths show a dark cavity → low mean pixel value + high variance
        texture_bonus = 0.0
        try:
            h, w = img.shape[:2]
            mx, my = int(mouth_center[0]), int(mouth_center[1])
            half_w = max(4, int(eye_dist * 0.15))
            half_h = max(3, int(eye_dist * 0.10))
            # Sample just below the mouth corner midpoint
            py1 = max(0, my)
            py2 = min(h, my + half_h * 2)
            px1 = max(0, mx - half_w)
            px2 = min(w, mx + half_w)
            if py2 > py1 and px2 > px1:
                patch = img[py1:py2, px1:px2]
                if patch.size > 0:
                    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
                    mean_val = float(np.mean(gray_patch))
                    std_val = float(np.std(gray_patch))
                    # Dark interior (< 80) with moderate variation → open mouth
                    if mean_val < 80 and std_val > 15:
                        texture_bonus = 0.08
                    elif mean_val < 100 and std_val > 20:
                        texture_bonus = 0.04
        except Exception:
            pass

        ratio = geom_ratio + texture_bonus
        is_open = ratio > threshold
        print(
            f"[ReActor V3]   Mouth (5-kps): nose_to_mouth={nose_to_mouth:.1f}px, "
            f"eye_dist={eye_dist:.1f}px, geom={geom_ratio:.3f}, tex_bonus={texture_bonus:.3f}, "
            f"ratio={ratio:.3f}, open={'YES' if is_open else 'no'} (thr={threshold:.2f})"
        )
        return ratio, is_open

    def _build_mouth_mask(self, face, image_shape: Tuple[int, int, int],
                          mouth_ratio: float = 0.0) -> Optional[np.ndarray]:
        """
        Build a soft feathered mask covering the mouth region.

        Uses detailed landmarks when available; falls back to 5-kps estimation.
        The mask size adapts to how open the mouth is (larger opening → bigger mask).

        Returns:
            Float32 mask [0-1] same H×W as image, or None on failure.
        """
        img_h, img_w = image_shape[:2]

        # ── Try detailed landmarks ────────────────────────────────────────
        landmark_2d = getattr(face, 'landmark_2d_106', None)
        if landmark_2d is None:
            landmark_2d = getattr(face, 'landmark_3d_68', None)
        if landmark_2d is None:
            landmark_2d = getattr(face, 'landmark_2d_68', None)

        mouth_pts = None
        if landmark_2d is not None:
            lm = np.asarray(landmark_2d, dtype=np.float32)
            if lm.ndim == 2:
                if lm.shape[0] >= 106:
                    # 106-pt: outer lip 52-71, inner 72-91 (approx range)
                    mouth_pts = lm[52:92, :2].copy()
                elif lm.shape[0] >= 68:
                    # 68-pt: outer lip 48-59, inner 60-67
                    mouth_pts = lm[48:68, :2].copy()

        # ── Fallback: estimate from 5 key-points ─────────────────────────
        if mouth_pts is None:
            kps = getattr(face, 'kps', None)
            if kps is None or len(kps) < 5:
                return None
            kps = np.asarray(kps, dtype=np.float32)
            mouth_l, mouth_r = kps[3], kps[4]
            nose = kps[2]
            eye_dist = float(np.linalg.norm(kps[1] - kps[0]))
            if eye_dist < 1.0:
                eye_dist = 1.0

            cx = (mouth_l[0] + mouth_r[0]) / 2.0
            cy = (mouth_l[1] + mouth_r[1]) / 2.0
            half_w = max(8, (mouth_r[0] - mouth_l[0]) / 2.0 * 1.3)
            # Vertical extent grows with mouth openness
            open_extra = max(0.0, mouth_ratio - 0.15) * eye_dist * 1.5
            half_h = max(6, eye_dist * 0.18 + open_extra)

            # Build an ellipse of sample points
            angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            mouth_pts = np.stack([
                cx + half_w * np.cos(angles),
                cy + half_h * np.sin(angles),
            ], axis=1).astype(np.float32)

        if mouth_pts is None or len(mouth_pts) < 3:
            return None

        # Expand slightly for safety margin
        centroid = mouth_pts.mean(axis=0)
        mouth_pts_expanded = centroid + (mouth_pts - centroid) * 1.25

        hull = cv2.convexHull(mouth_pts_expanded.astype(np.int32))
        mask = np.zeros((img_h, img_w), dtype=np.float32)
        cv2.fillConvexPoly(mask, hull, 1.0)

        # Dilate to cover surrounding skin that also gets distorted
        bbox = face.bbox if hasattr(face, 'bbox') else None
        if bbox is not None:
            face_w = max(1, int(bbox[2] - bbox[0]))
            face_h = max(1, int(bbox[3] - bbox[1]))
        else:
            face_w, face_h = img_w // 4, img_h // 4
        dilate_px = max(5, int(min(face_w, face_h) * 0.08))
        kernel = np.ones((dilate_px, dilate_px), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Feather the edges for seamless blending
        feather = max(11, int(min(face_w, face_h) * 0.15))
        if feather % 2 == 0:
            feather += 1
        mask = cv2.GaussianBlur(mask, (feather, feather), feather * 0.35)
        mask = np.clip(mask, 0.0, 1.0)

        return mask

    def _preserve_mouth_region(self,
                                original_img: np.ndarray,
                                swapped_img: np.ndarray,
                                target_face,
                                mouth_ratio: float = 0.0) -> np.ndarray:
        """
        When the target face has an open mouth, blend the original mouth region
        back into the swapped result to prevent pixelation / distortion.

        The blend strength is adaptive:
          - Slightly open  (ratio ~0.28-0.40): gentle blend (0.3-0.5)
          - Wide open       (ratio >0.50):     strong blend (up to full strength)

        This preserves the original mouth interior (teeth, tongue, cavity) while
        keeping the rest of the swapped face intact.
        """
        if not self.mouth_protect_enabled:
            return swapped_img
        if original_img is None or swapped_img is None or target_face is None:
            return swapped_img
        if original_img.shape != swapped_img.shape:
            return swapped_img

        mouth_mask = self._build_mouth_mask(target_face, swapped_img.shape, mouth_ratio)
        if mouth_mask is None:
            return swapped_img

        if float(np.max(mouth_mask)) < 0.01:
            return swapped_img

        # Adaptive strength: scale with how open the mouth is
        base_strength = self.mouth_protect_strength
        openness_factor = min(1.0, max(0.0, (mouth_ratio - self.mouth_open_threshold) / 0.25))
        # Ramp from ~40% of base at threshold to 100% at threshold+0.25
        effective_strength = base_strength * (0.4 + 0.6 * openness_factor)
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))

        # Blend original mouth back into swapped image
        orig_f = original_img.astype(np.float32)
        swap_f = swapped_img.astype(np.float32)
        mask_3ch = (mouth_mask * effective_strength)[:, :, None]

        blended = swap_f * (1.0 - mask_3ch) + orig_f * mask_3ch
        result = np.clip(blended, 0, 255).astype(np.uint8)

        coverage_pct = float(np.mean(mouth_mask > 0.05)) * 100
        print(
            f"[ReActor V3]   Mouth preserved: ratio={mouth_ratio:.3f}, "
            f"effective_strength={effective_strength:.2f}, "
            f"coverage={coverage_pct:.1f}% of image"
        )
        return result

    # ── End Mouth Preservation ────────────────────────────────────────────────

    def set_auto_face_fix(self, enabled: bool):
        """Enable/disable automatic face quality fix (match reference detail)."""
        self.auto_face_fix = bool(enabled)
        print(f"[ReActor V3] Auto face fix set to: {self.auto_face_fix}")

    def set_cleanup_mode(self, aggressive: bool):
        """Set whether cleanup should be aggressive (unload all models)"""
        self.aggressive_cleanup = aggressive
        print(f"[ReActor V3] Cleanup mode set to aggressive={aggressive}")

    def set_occlusion_handling(self, enabled: bool, strength: float = 1.0, sensitivity: float = 0.55):
        """Configure adaptive foreground occlusion preservation."""
        self.occlusion_enabled = bool(enabled)
        self.occlusion_strength = float(np.clip(strength, 0.0, 1.0))
        self.occlusion_sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        print(
            f"[ReActor V3] Occlusion handling set: enabled={self.occlusion_enabled}, "
            f"strength={self.occlusion_strength:.2f}, sensitivity={self.occlusion_sensitivity:.2f}"
        )
    
    def load_restorer(self, model_name: str) -> bool:
        if model_name == self.current_restorer_name and self.current_restorer is not None:
            print(f"[ReActor V3]   Restorer '{model_name}' already loaded (cached)")
            return True
        
        if not model_name or model_name.lower() == 'none':
            self.current_restorer = None
            self.current_restorer_name = None
            print(f"[ReActor V3]   Restorer: None (skipping restoration)")
            return True
        
        model_path = os.path.join(self.facerestore_path, model_name)
        if not os.path.exists(model_path):
            print(f"[ReActor V3]   ERROR: Restorer model not found: {model_path}")
            return False
        
        try:
            print(f"[ReActor V3]   Loading restorer: {model_name}")
            t0 = time.time()
            self.current_restorer = get_gpen_restorer(model_path, device='cuda')
            self.current_restorer_name = model_name
            elapsed = time.time() - t0
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"[ReActor V3]   Restorer loaded: {model_name} ({model_size_mb:.1f} MB) in {elapsed:.2f}s")
            return True
        except Exception as e:
            print(f"[ReActor V3]   ERROR loading restorer: {e}")
            return False

    def _get_safe_face_bbox(self, face, image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Clamp face bbox to image bounds and ensure a valid region."""
        img_h, img_w = image_shape[:2]

        if not hasattr(face, 'bbox') or face.bbox is None:
            return 0, 0, img_w, img_h

        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        return x1, y1, x2, y2

    def _build_soft_face_mask(self, face, image_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Create a feathered alpha mask around the swapped face region."""
        img_h, img_w = image_shape[:2]
        x1, y1, x2, y2 = self._get_safe_face_bbox(face, image_shape)

        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)

        # Expand region so blend transition happens outside key facial features.
        pad_x = max(4, int(face_w * 0.18))
        pad_y = max(4, int(face_h * 0.22))
        ex1 = max(0, x1 - pad_x)
        ey1 = max(0, y1 - pad_y)
        ex2 = min(img_w, x2 + pad_x)
        ey2 = min(img_h, y2 + pad_y)

        mask = np.zeros((img_h, img_w), dtype=np.float32)

        center = ((ex1 + ex2) // 2, (ey1 + ey2) // 2)
        axes = (max(2, (ex2 - ex1) // 2), max(2, (ey2 - ey1) // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        # Landmarks (if available) help keep high-confidence face areas in the mask.
        if hasattr(face, 'kps') and face.kps is not None:
            kps = np.asarray(face.kps, dtype=np.float32)
            if kps.ndim == 2 and kps.shape[0] >= 3:
                lm_mask = np.zeros_like(mask)
                hull = cv2.convexHull(kps.astype(np.int32))
                cv2.fillConvexPoly(lm_mask, hull, 1.0)
                dilate_k = max(3, int(min(face_w, face_h) * 0.10))
                kernel = np.ones((dilate_k, dilate_k), dtype=np.uint8)
                lm_mask = cv2.dilate(lm_mask, kernel, iterations=1)
                mask = np.maximum(mask, lm_mask * 0.9)

        feather = max(9, int(min(face_w, face_h) * 0.20))
        if feather % 2 == 0:
            feather += 1
        mask = cv2.GaussianBlur(mask, (feather, feather), feather * 0.35)
        mask = np.clip(mask, 0.0, 1.0)

        return mask, (x1, y1, x2, y2)

    def _build_adaptive_occlusion_mask(
        self,
        original_img: np.ndarray,
        processed_img: np.ndarray,
        face_mask: np.ndarray
    ) -> np.ndarray:
        """
        Build an adaptive mask for foreground objects that should stay in front of the face.
        Uses lost-edge + strong-change heuristics and keeps components that enter from face boundary.
        """
        if original_img is None or processed_img is None or face_mask is None:
            return np.zeros((0, 0), dtype=np.float32)

        if original_img.shape != processed_img.shape:
            return np.zeros(original_img.shape[:2], dtype=np.float32)

        h, w = original_img.shape[:2]
        if face_mask.shape[:2] != (h, w):
            return np.zeros((h, w), dtype=np.float32)

        sensitivity = float(np.clip(self.occlusion_sensitivity, 0.0, 1.0))
        strength = float(np.clip(self.occlusion_strength, 0.0, 1.0))
        if strength <= 0.0:
            return np.zeros((h, w), dtype=np.float32)

        roi_mask = (face_mask > 0.10)
        if not np.any(roi_mask):
            return np.zeros((h, w), dtype=np.float32)

        gray_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gray_proc = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_orig, gray_proc).astype(np.float32)

        roi_values = diff[roi_mask]
        if roi_values.size < 20:
            return np.zeros((h, w), dtype=np.float32)

        change_percentile = 80.0 - (sensitivity * 35.0)  # high sensitivity -> lower threshold
        thr_dynamic = float(np.percentile(roi_values, np.clip(change_percentile, 35.0, 90.0)))
        thr_floor = 10.0 + (1.0 - sensitivity) * 20.0
        strong_change = diff >= max(thr_dynamic, thr_floor)

        canny_low = int(35 + (1.0 - sensitivity) * 45)
        canny_high = int(90 + (1.0 - sensitivity) * 95)
        edges_orig = cv2.Canny(gray_orig, canny_low, canny_high)
        edges_proc = cv2.Canny(gray_proc, canny_low, canny_high)
        lost_edges = (edges_orig > 0) & (edges_proc == 0)

        hsv_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        hsv_proc = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
        sat_delta = cv2.absdiff(hsv_orig[:, :, 1], hsv_proc[:, :, 1]).astype(np.float32)
        val_delta = cv2.absdiff(hsv_orig[:, :, 2], hsv_proc[:, :, 2]).astype(np.float32)
        sat_thr = 12.0 + (1.0 - sensitivity) * 12.0
        val_thr = 20.0 + (1.0 - sensitivity) * 16.0
        chroma_shift = (sat_delta >= sat_thr) & (val_delta >= val_thr)

        candidate = (strong_change & (lost_edges | chroma_shift) & roi_mask).astype(np.uint8)
        if candidate.sum() == 0:
            return np.zeros((h, w), dtype=np.float32)

        face_binary = (face_mask > 0.10).astype(np.uint8)
        boundary_k = max(3, int(min(h, w) * 0.01))
        boundary_kernel = np.ones((boundary_k, boundary_k), dtype=np.uint8)
        inner = cv2.erode(face_binary, boundary_kernel, iterations=1)
        boundary = np.clip(face_binary - inner, 0, 1).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        filtered = np.zeros_like(candidate)
        min_area = max(14, int(face_binary.sum() * 0.002))
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            comp = (labels == label_id)
            # Occluders usually enter from outside face boundary.
            if np.any(boundary[comp] > 0):
                filtered[comp] = 1

        if filtered.sum() == 0:
            return np.zeros((h, w), dtype=np.float32)

        coverage = float(filtered.sum()) / float(max(1, face_binary.sum()))
        # If mask is too large, treat as false positive and skip.
        if coverage > 0.40:
            return np.zeros((h, w), dtype=np.float32)

        morph_k = max(3, int(min(h, w) * 0.008))
        morph_kernel = np.ones((morph_k, morph_k), dtype=np.uint8)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
        filtered = cv2.dilate(filtered, morph_kernel, iterations=1)

        feather = max(5, int(min(h, w) * 0.012))
        if feather % 2 == 0:
            feather += 1
        occ_mask = cv2.GaussianBlur(filtered.astype(np.float32), (feather, feather), feather * 0.25)
        occ_mask = np.clip(occ_mask, 0.0, 1.0)
        occ_mask = np.minimum(occ_mask, np.clip(face_mask * 1.15, 0.0, 1.0))
        occ_mask *= strength
        return occ_mask.astype(np.float32)

    def _preserve_foreground_occlusions(
        self,
        original_img: np.ndarray,
        processed_img: np.ndarray,
        target_face,
        stage: str = "post"
    ) -> np.ndarray:
        """Blend original pixels back where adaptive occlusion mask indicates foreground occluders."""
        if not self.occlusion_enabled:
            return processed_img
        if original_img is None or processed_img is None or target_face is None:
            return processed_img
        if original_img.shape != processed_img.shape:
            return processed_img

        face_mask, _ = self._build_soft_face_mask(target_face, processed_img.shape)
        occ_mask = self._build_adaptive_occlusion_mask(original_img, processed_img, face_mask)
        if occ_mask.size == 0 or float(np.max(occ_mask)) < 0.01:
            return processed_img

        processed_f = processed_img.astype(np.float32)
        original_f = original_img.astype(np.float32)
        occ_3ch = occ_mask[:, :, None]
        blended = processed_f * (1.0 - occ_3ch) + original_f * occ_3ch

        coverage = float(np.mean(occ_mask > 0.05))
        print(
            f"[ReActor V3] Occlusion preservation ({stage}): "
            f"coverage={coverage:.3f}, strength={self.occlusion_strength:.2f}"
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _texture_energy(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """Estimate local detail using Laplacian magnitude."""
        if img is None or img.size == 0:
            return 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        energy = np.abs(lap)

        if mask is not None:
            valid = mask > 0.25
            if np.any(valid):
                return float(np.mean(energy[valid]))

        return float(np.mean(energy))

    def _get_adaptive_restore_weight(self, swapped_img: np.ndarray, restored_img: np.ndarray, face_mask: np.ndarray) -> Tuple[float, float]:
        """
        Prevent over-sharp restored faces by adapting blend weight to local detail ratio.
        Returns (restore_weight, detail_ratio).
        """
        swapped_detail = self._texture_energy(swapped_img, face_mask)
        restored_detail = self._texture_energy(restored_img, face_mask)

        if swapped_detail <= 1e-6:
            return 0.85, 1.0

        detail_ratio = restored_detail / swapped_detail
        restore_weight = 1.0
        if detail_ratio > 1.08:
            # If restored face is much sharper than base image, pull it back slightly.
            restore_weight = max(0.65, 1.0 - 0.55 * (detail_ratio - 1.0))

        return float(np.clip(restore_weight, 0.65, 1.0)), float(detail_ratio)

    def _get_face_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Run face detection on an image and return the embedding of the largest face.
        Used for identity-aware restore weight computation.
        """
        if self.face_analyser is None:
            return None
        try:
            faces = self.face_analyser.get(img)
            if faces:
                best = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                    if hasattr(f, 'bbox') else 0
                )
                if hasattr(best, 'embedding') and best.embedding is not None:
                    return best.embedding
        except Exception as e:
            print(f"[ReActor V3] Could not extract embedding: {e}")
        return None

    def _harmonize_restored_face(self, swapped_img: np.ndarray, restored_img: np.ndarray,
                                  target_face, override_restore_weight: float = None) -> np.ndarray:
        """
        Blend restored face back into the swapped image with adaptive strength and seam cleanup.
        Uses identity-aware override weight when provided (Section B + F).
        """
        if swapped_img is None or restored_img is None:
            return restored_img
        if swapped_img.shape != restored_img.shape:
            return restored_img

        mask, (x1, y1, x2, y2) = self._build_soft_face_mask(target_face, swapped_img.shape)
        if mask is None:
            return restored_img

        swapped_f = swapped_img.astype(np.float32)
        restored_f = restored_img.astype(np.float32)

        # Use identity-aware weight if provided, otherwise fall back to texture-based
        if override_restore_weight is not None:
            restore_weight = override_restore_weight
            detail_ratio = 0.0  # Not computed when overridden
        else:
            restore_weight, detail_ratio = self._get_adaptive_restore_weight(swapped_img, restored_img, mask)

        harmonized = cv2.addWeighted(restored_f, restore_weight, swapped_f, 1.0 - restore_weight, 0.0)

        mask_3ch = mask[:, :, None]
        blended = harmonized * mask_3ch + swapped_f * (1.0 - mask_3ch)

        # Extra seam protection: reintroduce a little swapped image on the transition ring.
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        ring_k = max(3, int(min(face_w, face_h) * 0.08))
        ring_kernel = np.ones((ring_k, ring_k), dtype=np.uint8)

        binary_mask = (mask * 255.0).astype(np.uint8)
        inner = cv2.erode(binary_mask, ring_kernel, iterations=1).astype(np.float32) / 255.0
        outer = cv2.dilate(binary_mask, ring_kernel, iterations=1).astype(np.float32) / 255.0
        edge_ring = np.clip(outer - inner, 0.0, 1.0)[:, :, None]

        seam_preserve = 0.40
        blended = blended * (1.0 - edge_ring * seam_preserve) + swapped_f * (edge_ring * seam_preserve)

        print(
            f"[ReActor V3] Harmonized restore: detail_ratio={detail_ratio:.2f}, "
            f"restore_weight={restore_weight:.2f}, seam_preserve={seam_preserve:.2f}"
        )

        return np.clip(blended, 0, 255).astype(np.uint8)
    
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
            print("")
            print("[ReActor V3] ════════════════════════════════════════")
            print("[ReActor V3]   FACE SWAP PIPELINE START")
            print("[ReActor V3] ════════════════════════════════════════")
            print(f"[ReActor V3]   Source image: {source_img.shape[1]}x{source_img.shape[0]} ({source_img.dtype})")
            print(f"[ReActor V3]   Target image: {target_img.shape[1]}x{target_img.shape[0]} ({target_img.dtype})")
            print(f"[ReActor V3]   Gender match: {gender_match}")
            print(f"[ReActor V3]   Restore model: {restore_model or 'None'}")
            print(f"[ReActor V3]   Source face index: {source_face_index}, Target face index: {target_face_index}")
            pipeline_start = time.time()
            
            # Initialize
            if self.face_analyser is None:
                self.initialize_face_analyser()
            if self.face_swapper is None:
                self.initialize_face_swapper()
            
            # Detect faces
            print("[ReActor V3] ── Detecting Faces ──")
            print("[ReActor V3]   Scanning source image...")
            source_faces = self.get_faces(source_img)
            print("[ReActor V3]   Scanning target image...")
            target_faces = self.get_faces(target_img)
            
            if not source_faces:
                print("[ReActor V3]   ERROR: No face detected in source image")
                return target_img, "Error: No face in source"
            if not target_faces:
                print("[ReActor V3]   ERROR: No face detected in target image")
                return target_img, "Error: No face in target"
            
            print(f"[ReActor V3]   Source faces: {len(source_faces)}, Target faces: {len(target_faces)}")
            
            # Apply gender matching
            print("[ReActor V3] ── Gender Matching ──")
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
            print("[ReActor V3] ── Swapping Face ──")
            print(f"[ReActor V3]   Source face index: {source_face_index}, Target face index: {target_face_index}")
            print(f"[ReActor V3]   Target image size: {target_img.shape[1]}x{target_img.shape[0]}")
            if hasattr(source_face, 'embedding') and source_face.embedding is not None:
                emb_norm = float(np.linalg.norm(source_face.embedding))
                print(f"[ReActor V3]   Source face embedding norm: {emb_norm:.2f}")
            if hasattr(target_face, 'bbox'):
                bbox = [int(v) for v in target_face.bbox]
                face_w = bbox[2] - bbox[0]
                face_h = bbox[3] - bbox[1]
                face_area_pct = (face_w * face_h) / (target_img.shape[0] * target_img.shape[1]) * 100
                print(f"[ReActor V3]   Target face bbox: {bbox} ({face_w}x{face_h}px, {face_area_pct:.1f}% of image)")
            t0 = time.time()
            result = self.face_swapper.get(target_img, target_face, source_face, paste_back=True)
            swap_elapsed = time.time() - t0
            print(f"[ReActor V3]   Face swap completed in {swap_elapsed:.3f}s")
            
            # Mouth preservation: detect open mouth → blend original mouth back
            if self.mouth_protect_enabled:
                mouth_ratio, mouth_is_open = self._detect_mouth_open(target_face, target_img)
                if mouth_is_open:
                    print(f"[ReActor V3]   ⚠ Open mouth detected (ratio={mouth_ratio:.3f}) — preserving mouth region")
                    result = self._preserve_mouth_region(target_img, result, target_face, mouth_ratio)
            
            # Compute swap difference stats
            diff = cv2.absdiff(target_img, result)
            mean_diff = float(np.mean(diff))
            max_diff = float(np.max(diff))
            print(f"[ReActor V3]   Swap diff stats: mean={mean_diff:.2f}, max={max_diff:.0f}")
            
            result = self._preserve_foreground_occlusions(target_img, result, target_face, stage="post-swap")
            
            # Restore using GPEN if specified
            if restore_model:
                if not self.load_restorer(restore_model):
                    return result, "Swapped (restoration failed to load)"
                
                if self.current_restorer:
                    print(f"[ReActor V3] ── Restoring with {restore_model} ──")
                    if torch.cuda.is_available():
                        vram_before = torch.cuda.memory_allocated() / (1024**3)
                        print(f"[ReActor V3]   VRAM before restore: {vram_before:.2f} GB")
                    
                    t0_restore = time.time()
                    restored_result = self.current_restorer.restore(result)
                    restore_elapsed = time.time() - t0_restore
                    print(f"[ReActor V3]   GPEN restoration completed in {restore_elapsed:.3f}s")
                    
                    # Compare swapped vs restored to show what restoration changed
                    restore_diff = cv2.absdiff(result, restored_result)
                    restore_mean_diff = float(np.mean(restore_diff))
                    print(f"[ReActor V3]   Restoration diff: mean_change={restore_mean_diff:.2f}")
                    
                    # Color analysis: compare face region colors before/after restore
                    if hasattr(target_face, 'bbox'):
                        bx1, by1, bx2, by2 = [int(v) for v in target_face.bbox]
                        bx1, by1 = max(0, bx1), max(0, by1)
                        bx2, by2 = min(result.shape[1], bx2), min(result.shape[0], by2)
                        face_before = result[by1:by2, bx1:bx2]
                        face_after = restored_result[by1:by2, bx1:bx2]
                        if face_before.size > 0 and face_after.size > 0:
                            # BGR mean colors
                            mean_before = np.mean(face_before, axis=(0,1))
                            mean_after = np.mean(face_after, axis=(0,1))
                            color_shift = np.abs(mean_after - mean_before)
                            print(f"[ReActor V3]   Face color (BGR) before restore: [{mean_before[0]:.1f}, {mean_before[1]:.1f}, {mean_before[2]:.1f}]")
                            print(f"[ReActor V3]   Face color (BGR) after restore:  [{mean_after[0]:.1f}, {mean_after[1]:.1f}, {mean_after[2]:.1f}]")
                            print(f"[ReActor V3]   Color shift (BGR): [{color_shift[0]:.1f}, {color_shift[1]:.1f}, {color_shift[2]:.1f}]")
                            
                            # Also check body vs face color mismatch
                            # Sample body region (below face)
                            body_y_start = min(by2, result.shape[0] - 1)
                            body_y_end = min(by2 + (by2 - by1), result.shape[0])
                            if body_y_end > body_y_start:
                                body_region = restored_result[body_y_start:body_y_end, bx1:bx2]
                                if body_region.size > 0:
                                    mean_body = np.mean(body_region, axis=(0,1))
                                    face_body_diff = np.abs(mean_after - mean_body)
                                    print(f"[ReActor V3]   Body color (BGR) below face:   [{mean_body[0]:.1f}, {mean_body[1]:.1f}, {mean_body[2]:.1f}]")
                                    print(f"[ReActor V3]   Face-vs-Body color diff (BGR): [{face_body_diff[0]:.1f}, {face_body_diff[1]:.1f}, {face_body_diff[2]:.1f}]")
                                    total_mismatch = float(np.mean(face_body_diff))
                                    if total_mismatch > 15:
                                        print(f"[ReActor V3]   ⚠ COLOR MISMATCH DETECTED: avg diff={total_mismatch:.1f} (threshold=15)")
                                    else:
                                        print(f"[ReActor V3]   ✓ Color consistency OK: avg diff={total_mismatch:.1f}")
                    
                    # ── Identity-Aware + Resolution-Aware Restore Weight (Sections B, F) ──
                    print(f"[ReActor V3] ── Computing Identity-Aware Restore Weight ──")
                    source_emb = source_face.embedding if hasattr(source_face, 'embedding') else None
                    swapped_emb = self._get_face_embedding(result)
                    restored_emb = self._get_face_embedding(restored_result)

                    identity_weight = compute_identity_restore_weight(
                        source_emb, swapped_emb, restored_emb, base_restore_weight=0.8
                    )
                    res_limit = compute_resolution_restore_limit(
                        target_face.bbox if hasattr(target_face, 'bbox') else None
                    )
                    effective_weight = min(identity_weight, res_limit)
                    print(f"[ReActor V3]   Effective restore weight: {effective_weight:.3f} "
                          f"(identity={identity_weight:.3f}, res_limit={res_limit:.2f})")

                    print(f"[ReActor V3] ── Harmonizing Restored Face ──")
                    t0_harm = time.time()
                    restored_result = self._harmonize_restored_face(
                        result, restored_result, target_face,
                        override_restore_weight=effective_weight,
                    )
                    harm_elapsed = time.time() - t0_harm
                    print(f"[ReActor V3]   Harmonization completed in {harm_elapsed:.3f}s")
                    
                    restored_result = self._preserve_foreground_occlusions(
                        target_img, restored_result, target_face, stage="post-restore"
                    )
                    
                    # ── Adaptive Face Enhancement (replaces static face fix) ──
                    if self.auto_face_fix:
                        print("[ReActor V3] ── Adaptive Face Enhancement ──")
                        t0_fix = time.time()
                        restored_result = auto_fix_face(
                            reference_img=source_img,
                            output_img=restored_result,
                            face_analyser=self.face_analyser,
                            target_img=target_img,
                            source_face=source_face,
                        )
                        fix_elapsed = time.time() - t0_fix
                        print(f"[ReActor V3]   Adaptive face enhancement completed in {fix_elapsed:.3f}s")

                    # Final quality metrics
                    if hasattr(target_face, 'bbox'):
                        bx1, by1, bx2, by2 = [int(v) for v in target_face.bbox]
                        bx1, by1 = max(0, bx1), max(0, by1)
                        bx2, by2 = min(restored_result.shape[1], bx2), min(restored_result.shape[0], by2)
                        final_face = restored_result[by1:by2, bx1:bx2]
                        if final_face.size > 0:
                            gray_face = cv2.cvtColor(final_face, cv2.COLOR_BGR2GRAY)
                            laplacian_var = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
                            brightness = float(np.mean(gray_face))
                            contrast = float(np.std(gray_face))
                            print(f"[ReActor V3] ── Final Face Quality Metrics ──")
                            print(f"[ReActor V3]   Sharpness (Laplacian var): {laplacian_var:.1f}")
                            print(f"[ReActor V3]   Brightness: {brightness:.1f}/255")
                            print(f"[ReActor V3]   Contrast (std): {contrast:.1f}")
                    
                    if torch.cuda.is_available():
                        vram_after = torch.cuda.memory_allocated() / (1024**3)
                        print(f"[ReActor V3]   VRAM after restore: {vram_after:.2f} GB")
                    
                    print(f"[ReActor V3] ✓ Swapped and restored with {restore_model}")
                    pipeline_elapsed = time.time() - pipeline_start
                    print(f"[ReActor V3] ── Total pipeline time: {pipeline_elapsed:.3f}s ──")
                    print("[ReActor V3] ════════════════════════════════════════")
                    print("")
                    # Automatic VRAM cleanup after processing
                    if self.auto_cleanup:
                        self.cleanup_memory(aggressive=self.aggressive_cleanup)
                    return restored_result, f"Swapped and restored with {restore_model}"
            
            # Automatic VRAM cleanup after processing
            if self.auto_cleanup:
                self.cleanup_memory(aggressive=self.aggressive_cleanup)
            print(f"[ReActor V3] ✓ Face swapped successfully (no restoration)")
            pipeline_elapsed = time.time() - pipeline_start
            print(f"[ReActor V3] ── Total pipeline time: {pipeline_elapsed:.3f}s ──")
            print("[ReActor V3] ════════════════════════════════════════")
            print("")
            return result, "Face swapped successfully"
            
        except Exception as e:
            print(f"[ReActor V3] ✗ ERROR: {e}")
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
        print(f"[ReActor V3] ── Memory Cleanup (aggressive={aggressive}) ──")
        
        if torch.cuda.is_available():
            vram_before_alloc = torch.cuda.memory_allocated() / (1024**3)
            vram_before_res = torch.cuda.memory_reserved() / (1024**3)
            print(f"[ReActor V3]   VRAM before cleanup - Allocated: {vram_before_alloc:.2f} GB, Reserved: {vram_before_res:.2f} GB")
        
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
                freed_alloc = vram_before_alloc - memory_allocated if 'vram_before_alloc' in dir() else 0
                print(f"[ReActor V3]   VRAM after cleanup - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
                print(f"[ReActor V3]   VRAM freed: ~{max(0, freed_alloc):.2f} GB")
            print(f"[ReActor V3]   Cleanup complete")
            
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
