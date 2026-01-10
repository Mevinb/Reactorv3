"""
ReActor V3 - Face Detection and Alignment Utilities

This module handles:
- Face detection using InsightFace
- Precise facial landmark alignment for GPEN processing
- Face cropping at high resolution (512/1024)
- Similarity transformations for proper feature alignment
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
import os


def estimate_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transformation matrix from source to destination points.
    
    This is critical for GPEN as it expects faces aligned to a specific template
    where eyes are at fixed positions. Misalignment causes severe distortion.
    
    Args:
        src_pts: Source landmarks (N, 2)
        dst_pts: Destination landmarks (N, 2)
        
    Returns:
        Transformation matrix (2, 3)
    """
    # Use OpenCV's estimateAffinePartial2D for robust estimation
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    return M


def get_alignment_matrix(landmarks: np.ndarray, target_size: int = 512, 
                        scale_factor: float = 1.0) -> np.ndarray:
    """
    Calculate affine transformation matrix to align face to standard template.
    
    The standard template positions eyes horizontally centered with specific
    spacing, which is what GPEN's StyleGAN prior expects.
    
    Args:
        landmarks: Facial keypoints (5, 2) - left_eye, right_eye, nose, left_mouth, right_mouth
        target_size: Output resolution (512 or 1024)
        scale_factor: Adjustment for face crop size (default 1.0)
        
    Returns:
        Affine transformation matrix (2, 3)
    """
    # Standard ArcFace template at 512x512 resolution
    # These are the "ideal" positions for facial landmarks
    arcface_template_512 = np.array([
        [192.98138, 239.94708],  # left eye
        [318.90277, 240.19360],  # right eye
        [256.63416, 314.01935],  # nose tip
        [201.26117, 371.41043],  # left mouth corner
        [313.08905, 371.15118]   # right mouth corner
    ], dtype=np.float32)
    
    # Scale template to target resolution
    if target_size != 512:
        scale = target_size / 512.0
        template = arcface_template_512 * scale
    else:
        template = arcface_template_512.copy()
    
    # Apply additional scale factor if needed
    if scale_factor != 1.0:
        center = np.array([target_size / 2, target_size / 2])
        template = (template - center) * scale_factor + center
    
    # Estimate similarity transform
    M = estimate_similarity_transform(landmarks, template)
    
    return M


def align_and_crop_face(img: np.ndarray, landmarks: np.ndarray, 
                       target_size: int = 512) -> Optional[np.ndarray]:
    """
    Align and crop a face from an image using facial landmarks.
    
    This is THE critical function for GPEN quality. By extracting at the target
    resolution directly from the source image, we preserve maximum detail.
    
    Args:
        img: Source image (H, W, 3) in BGR
        landmarks: Facial keypoints (5, 2)
        target_size: Output resolution (512 or 1024)
        
    Returns:
        Aligned and cropped face (target_size, target_size, 3) or None
    """
    if landmarks is None or len(landmarks) < 5:
        return None
    
    try:
        # Calculate alignment matrix
        M = get_alignment_matrix(landmarks, target_size)
        
        if M is None:
            return None
        
        # Apply affine transformation
        # INTER_LINEAR is faster, INTER_CUBIC is higher quality
        aligned_face = cv2.warpAffine(
            img, 
            M, 
            (target_size, target_size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return aligned_face
        
    except Exception as e:
        print(f"[ReActor V3] Error in face alignment: {e}")
        return None


def get_face_bbox(landmarks: np.ndarray, margin: float = 0.3) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box from facial landmarks with margin.
    
    Args:
        landmarks: Facial keypoints (N, 2)
        margin: Margin factor around face (0.3 = 30% padding)
        
    Returns:
        (x1, y1, x2, y2) bounding box coordinates
    """
    x_min, y_min = landmarks.min(axis=0)
    x_max, y_max = landmarks.max(axis=0)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Add margin
    x1 = int(x_min - width * margin)
    y1 = int(y_min - height * margin)
    x2 = int(x_max + width * margin)
    y2 = int(y_max + height * margin)
    
    return x1, y1, x2, y2


def paste_face_back(background: np.ndarray, face: np.ndarray, 
                    landmarks: np.ndarray, target_size: int = 512,
                    blend: bool = True) -> np.ndarray:
    """
    Paste a restored face back into the original image with proper alignment.
    
    Args:
        background: Original image (H, W, 3)
        face: Restored face (target_size, target_size, 3)
        landmarks: Original facial landmarks (5, 2)
        target_size: Size of the restored face (512 or 1024)
        blend: Whether to apply seamless blending
        
    Returns:
        Image with face pasted back
    """
    print(f"[ReActor V3] paste_face_back called - background: {background.shape}, face: {face.shape}, target_size: {target_size}")
    result = background.copy()
    
    try:
        # Get the transformation matrix (same as used for extraction)
        M = get_alignment_matrix(landmarks, target_size)
        
        if M is None:
            print(f"[ReActor V3] ERROR: Could not get alignment matrix")
            return background
        
        # Calculate inverse transformation to map restored face back
        M_inv = cv2.invertAffineTransform(M)
        
        # Warp the restored face back to original position
        h, w = background.shape[:2]
        face_warped = cv2.warpAffine(
            face,
            M_inv,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Create mask for blending - use a LARGE feathered square
        # that covers the entire valid warped region
        mask_temp = np.ones((target_size, target_size), dtype=np.float32) * 255
        
        # Create a feather/fade at edges to blend smoothly
        # This prevents hard edges while avoiding black borders
        if blend:
            # Create a gradient mask from center outward
            center = target_size // 2
            y, x = np.ogrid[:target_size, :target_size]
            
            # Distance from center
            dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
            max_dist = center
            
            # Create gradient: 1.0 at center, fading to 0 at edges
            # Start fade at 70% radius to keep core solid
            fade_start = max_dist * 0.7
            mask_temp = np.clip((max_dist - dist_from_center) / (max_dist - fade_start), 0, 1)
            
            # Apply strong gaussian blur for seamless blending
            mask_temp = cv2.GaussianBlur(mask_temp, (0, 0), target_size * 0.05)
        
        # Warp the mask back to original position
        mask = cv2.warpAffine(
            mask_temp,
            M_inv,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Expand mask to 3 channels for blending
        mask_3ch = np.expand_dims(mask, axis=2)
        
        # Blend the face back using proper alpha compositing
        # Only blend where mask > 0 to avoid black borders
        result = background.copy().astype(np.float32)
        face_float = face_warped.astype(np.float32)
        
        # Alpha blend: result = foreground * alpha + background * (1 - alpha)
        result = face_float * mask_3ch + result * (1 - mask_3ch)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        print(f"[ReActor V3] paste_face_back completed successfully")
        return result
        
    except Exception as e:
        print(f"[ReActor V3] Error pasting face back: {e}")
        import traceback
        traceback.print_exc()
        return background


def smart_resolution_selector(bbox_size: int) -> int:
    """
    Intelligently select GPEN resolution based on face size.
    
    Using 1024 on small faces wastes computation and may introduce artifacts.
    This function optimizes the choice based on the actual face size.
    
    Args:
        bbox_size: Approximate size of the face bounding box in pixels
        
    Returns:
        Recommended resolution (512 or 1024)
    """
    # If face is smaller than 384px, use 512
    # If face is 384px or larger, use 1024 for maximum detail
    if bbox_size < 384:
        return 512
    else:
        return 1024


def calculate_face_size(landmarks: np.ndarray) -> int:
    """
    Calculate approximate face size from landmarks.
    
    Args:
        landmarks: Facial keypoints (5, 2)
        
    Returns:
        Approximate face size in pixels
    """
    if landmarks is None or len(landmarks) < 2:
        return 0
    
    # Calculate inter-eye distance as a proxy for face size
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    eye_distance = np.linalg.norm(left_eye - right_eye)
    
    # A typical face is about 3x the inter-eye distance
    face_size = int(eye_distance * 3)
    
    return face_size


class FaceAlignment:
    """
    High-level face alignment manager for ReActor V3.
    
    This class provides a clean interface for face operations needed
    by the swapping pipeline.
    """
    
    def __init__(self, default_resolution: int = 512):
        """
        Initialize face alignment manager.
        
        Args:
            default_resolution: Default resolution for face operations
        """
        self.default_resolution = default_resolution
        
    def process_face(self, img: np.ndarray, face_obj, 
                    auto_resolution: bool = True) -> Tuple[Optional[np.ndarray], int]:
        """
        Extract and align a face for restoration.
        
        Args:
            img: Source image
            face_obj: InsightFace face detection object with 'kps' attribute
            auto_resolution: Whether to automatically select resolution
            
        Returns:
            (aligned_face, resolution) tuple
        """
        if not hasattr(face_obj, 'kps') or face_obj.kps is None:
            return None, self.default_resolution
        
        landmarks = face_obj.kps.astype(np.float32)
        
        # Determine optimal resolution
        if auto_resolution:
            face_size = calculate_face_size(landmarks)
            resolution = smart_resolution_selector(face_size)
        else:
            resolution = self.default_resolution
        
        # Extract and align face
        aligned_face = align_and_crop_face(img, landmarks, resolution)
        
        return aligned_face, resolution
    
    def paste_back(self, background: np.ndarray, restored_face: np.ndarray,
                   face_obj, resolution: int) -> np.ndarray:
        """
        Paste a restored face back into the image.
        
        Args:
            background: Original image
            restored_face: Restored face from GPEN
            face_obj: Original InsightFace detection
            resolution: Resolution of the restored face
            
        Returns:
            Result image with face pasted back
        """
        if not hasattr(face_obj, 'kps') or face_obj.kps is None:
            return background
        
        landmarks = face_obj.kps.astype(np.float32)
        return paste_face_back(background, restored_face, landmarks, resolution, blend=True)
