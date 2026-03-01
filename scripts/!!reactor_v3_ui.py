"""
ReActor V3 - WebUI Forge Script Integration

This script integrates ReActor V3 into img2img and txt2img tabs as a post-processing
script that automatically enhances faces after SD generation using GPEN-512/1024.
"""

import os
import sys
import time
import gradio as gr
import modules.scripts as scripts
from typing import Optional
from modules import images
from modules.processing import Processed
from modules.shared import opts, state
from PIL import Image
import numpy as np
import cv2
import torch

# Add extension scripts path for imports
_ext_scripts_path = os.path.dirname(os.path.abspath(__file__))
if _ext_scripts_path not in sys.path:
    sys.path.insert(0, _ext_scripts_path)

# Import the SIMPLIFIED version
from reactor_v3_swapper_new import get_reactor_v3_engine

# Import adaptive pipeline
from reactor_v3_adaptive import (
    AdaptiveReActorPipeline,
    AdaptiveParams,
    CONFIDENCE_REVIEW_THRESHOLD,
)

# Global adaptive pipeline instance  (one per engine)
_adaptive_pipeline: Optional['AdaptiveReActorPipeline'] = None

def get_adaptive_pipeline(engine) -> 'AdaptiveReActorPipeline':
    global _adaptive_pipeline
    if _adaptive_pipeline is None or _adaptive_pipeline.engine is not engine:
        _adaptive_pipeline = AdaptiveReActorPipeline(engine)
    return _adaptive_pipeline


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR format"""
    if pil_img is None:
        return None
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL Image"""
    if cv2_img is None:
        return None
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def get_available_models():
    """Get list of available GPEN models"""
    try:
        # Get the WebUI root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        extension_dir = os.path.dirname(current_dir)
        extensions_dir = os.path.dirname(extension_dir)
        webui_dir = os.path.dirname(extensions_dir)
        
        # Use shared WebUI models directory
        models_path = os.path.join(webui_dir, 'models')
        engine = get_reactor_v3_engine(models_path)
        return engine.get_available_restorers()
    except Exception as e:
        print(f"[ReActor V3] Error getting models: {e}")
        import traceback
        traceback.print_exc()
        return ['None']


class ReactorV3Script(scripts.Script):
    """
    ReActor V3 - High-fidelity face swapping with GPEN restoration
    
    Automatically processes generated images to swap and enhance faces
    using GPEN-BFR-512 or GPEN-BFR-1024 models.
    """
    
    def title(self):
        return "ReActor V3 (GPEN-512/1024)"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        """Create UI elements that appear in img2img/txt2img tabs"""
        
        with gr.Accordion("ReActor V3 - GPEN High-Fidelity Face Swap", open=False):
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enable ReActor V3",
                    value=False,
                    info="Automatically swap and enhance faces after generation"
                )
            
            gr.Markdown("""
            **Post-Processing Face Enhancement** - Swap faces and restore at 512x512 or 1024x1024 resolution
            """)
            
            with gr.Row():
                with gr.Column():
                    source_image = gr.Image(
                        label="Source Face (face to copy)",
                        type="pil",
                        interactive=True
                    )
                    source_face_index = gr.Slider(
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=0,
                        label="Source Face Index",
                        info="Which face to use if multiple detected"
                    )
                
                with gr.Column():
                    target_face_index = gr.Slider(
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=0,
                        label="Target Face Index",
                        info="Which face to replace in generated image"
                    )
                    
                    restore_model = gr.Dropdown(
                        label="GPEN Restoration Model",
                        choices=get_available_models(),
                        value="None",
                        info="512 = fast, 1024 = ultra-quality"
                    )
            
            with gr.Row():
                gender_match = gr.Radio(
                    label="Gender Matching Mode",
                    choices=[
                        ("All (No Filter)", "A"),
                        ("Smart Match (Auto-detect)", "S"),
                        ("Male Only", "M"),
                        ("Female Only", "F")
                    ],
                    value="S",
                    info="S=Match source gender automatically, M/F=Filter specific gender"
                )
            
            with gr.Row():
                auto_resolution = gr.Checkbox(
                    label="Auto-Select Resolution",
                    value=True,
                    info="Automatically choose 512 or 1024 based on face size"
                )
                
                aggressive_cleanup = gr.Checkbox(
                    label="Aggressive Memory Cleanup",
                    value=True,
                    info="Clear model cache after each image (recommended for <12GB VRAM)"
                )
                
                refresh_button = gr.Button("🔄 Refresh Models")

            with gr.Row():
                occlusion_enabled = gr.Checkbox(
                    label="Adaptive Occlusion Handling",
                    value=True,
                    info="Keep foreground occluders in front of swapped faces"
                )
                occlusion_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=1.0,
                    label="Occlusion Preserve Strength",
                    info="How strongly original occluding objects are preserved"
                )
                occlusion_sensitivity = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.55,
                    label="Occlusion Detection Sensitivity",
                    info="Higher sensitivity catches more potential occluders"
                )

            with gr.Row():
                mouth_protect_enabled = gr.Checkbox(
                    label="Mouth Protection",
                    value=True,
                    info="Preserve original mouth region when target has open mouth (prevents pixelation)"
                )
                mouth_protect_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.75,
                    label="Mouth Protect Strength",
                    info="How strongly to blend original mouth back (higher = more original mouth kept)"
                )
                mouth_open_threshold = gr.Slider(
                    minimum=0.05,
                    maximum=0.60,
                    step=0.01,
                    value=0.28,
                    label="Mouth Open Threshold",
                    info="Mouth-open ratio above which protection activates (lower = trigger earlier)"
                )

            with gr.Row():
                auto_face_fix = gr.Checkbox(
                    label="\u2728 Auto Face Detail Fix",
                    value=True,
                    info="Automatically match output face sharpness/texture to reference — no manual tuning needed"
                )

            # ── Auto Face Match block ─────────────────────────────────────────
            with gr.Accordion("🎯 Auto Face Match (Embedding-Based)", open=False):
                gr.Markdown("""
                **Automatic face detection & matching** using ArcFace embedding cosine similarity.
                No more manual face index selection — the system automatically finds which target face
                best matches each source face and swaps only the correct ones.
                """)

                with gr.Row():
                    auto_match_enabled = gr.Checkbox(
                        label="Enable Auto Face Match",
                        value=True,
                        info="Automatically detect and match faces by embedding similarity (no manual index needed)"
                    )
                    auto_match_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=0.8,
                        step=0.05,
                        value=0.20,
                        label="Match Similarity Threshold",
                        info="Minimum cosine similarity to accept a match (lower = more permissive, 0.3+ = strict)"
                    )

                with gr.Row():
                    with gr.Column():
                        source_image_2 = gr.Image(
                            label="Additional Source Face 2 (optional)",
                            type="pil",
                            interactive=True
                        )
                    with gr.Column():
                        source_image_3 = gr.Image(
                            label="Additional Source Face 3 (optional)",
                            type="pil",
                            interactive=True
                        )

                gr.Markdown("""
                **How it works:**
                - Detects ALL faces in source image(s) and target image
                - Computes 512-dim ArcFace embedding for each face
                - Calculates cosine similarity between every source↔target pair
                - Uses Hungarian algorithm for globally optimal 1:1 assignment
                - Swaps only matched pairs above the similarity threshold
                
                **Use cases:**
                - **1 source, multi-target**: Only the best-matching target face gets swapped
                - **Multi-source, multi-target**: Each source auto-matched to its best target
                - **Additional sources**: Upload extra face images for multi-person swapping
                
                **Threshold guide:** 0.0 = always match, 0.2 = permissive, 0.4 = moderate, 0.6+ = strict same-person only
                """)

            # ── Adaptive Pipeline block ──────────────────────────────────────
            with gr.Accordion("⚙️ Adaptive Pipeline (Auto-Tune)", open=False):
                with gr.Row():
                    adaptive_enabled = gr.Checkbox(
                        label="Enable Adaptive Pipeline",
                        value=False,
                        info="Analyse face quality, auto-pick settings, detect artefacts and retry automatically"
                    )
                    adaptive_max_retries = gr.Slider(
                        minimum=0, maximum=2, step=1, value=1,
                        label="Max Adaptive Retries",
                        info="Extra passes when artefacts are detected (0–2)"
                    )

                with gr.Row():
                    adaptive_confidence_threshold = gr.Slider(
                        minimum=0.10, maximum=0.60, step=0.05, value=0.30,
                        label="Confidence Threshold (flag below)",
                        info="Images below this quality score are flagged for manual review instead of forced through"
                    )
                    adaptive_color_match = gr.Checkbox(
                        label="Force Colour Match",
                        value=False,
                        info="Always apply scene colour normalisation (overrides auto-detect)"
                    )

                with gr.Row():
                    adaptive_swap_strength = gr.Slider(
                        minimum=0.30, maximum=1.00, step=0.05, value=1.00,
                        label="Manual Swap Strength Override (0 = auto)",
                        info="0 = fully automatic; any other value locks the swap blend"
                    )
                    adaptive_restore_strength = gr.Slider(
                        minimum=0.00, maximum=1.00, step=0.05, value=0.00,
                        label="Manual Restore Strength Override (0 = auto)",
                        info="0 = fully automatic; >0 locks GPEN blend strength"
                    )

                with gr.Row():
                    adaptive_sharpen = gr.Slider(
                        minimum=0.00, maximum=1.00, step=0.05, value=0.00,
                        label="Manual Sharpen Override (0 = auto)",
                        info="0 = fully automatic"
                    )
                    adaptive_texture_blend = gr.Slider(
                        minimum=0.00, maximum=0.60, step=0.05, value=0.00,
                        label="Texture Preserve Blend (0 = auto)",
                        info="Re-add original high-frequency detail after restoration to prevent plastic skin"
                    )

                gr.Markdown("""
                **Adaptive Pipeline mode:**
                - **Auto**: Analyses source face → picks swap/restore/sharpen strengths automatically → detects plastic skin / grain / seam artefacts → retries with corrected params (up to Max Retries).
                - **Confidence Threshold**: If quality score is too low the result is NOT forced — the original image is returned and flagged in console.
                - Manual override sliders only apply when set above 0; otherwise the adaptive engine decides.
                """)

            # ────────────────────────────────────────────────────────────────

            gr.Markdown("""
            **💡 Tips:**
            - Upload source face image above
            - Enable checkbox to process generated images
            - GPEN-512: Fast, good quality (full body shots)
            - GPEN-1024: Slower, maximum quality (portraits)
            - Auto-resolution recommended for best speed/quality balance
            - **Auto Face Match**: Enable to skip manual face index — automatic embedding-based matching!
            - **Smart Match**: Automatically swaps only matching gender (e.g., female source → female target)
            - **Gender Filter**: Use M/F to swap only male or female faces regardless of source
            - **Aggressive Cleanup**: Enable if you have limited VRAM (<12GB) for consistent speed
            - **Multi-person swap**: Upload additional source faces in Auto Face Match section            - **Auto Face Detail Fix**: Compares reference face detail to output and auto-corrects sharpness/texture            
            **📁 Model Location:** `extensions/sd-webui-reactor-v3/models/facerestore_models/`
            """)
            
            # Refresh button handler
            refresh_button.click(
                fn=lambda: gr.Dropdown.update(choices=get_available_models()),
                inputs=[],
                outputs=[restore_model]
            )
        
        return [
            enabled, source_image, source_face_index, target_face_index,
            restore_model, gender_match, auto_resolution, aggressive_cleanup,
            occlusion_enabled, occlusion_strength, occlusion_sensitivity,
            # mouth protection controls
            mouth_protect_enabled, mouth_protect_strength, mouth_open_threshold,
            # auto face fix
            auto_face_fix,
            # auto face match controls
            auto_match_enabled, auto_match_threshold,
            source_image_2, source_image_3,
            # adaptive controls
            adaptive_enabled, adaptive_max_retries, adaptive_confidence_threshold,
            adaptive_color_match, adaptive_swap_strength, adaptive_restore_strength,
            adaptive_sharpen, adaptive_texture_blend,
        ]
    
    def postprocess_image(self, p, pp,
                         # base params
                         enabled, source_image, source_face_index,
                         target_face_index, restore_model, gender_match,
                         auto_resolution, aggressive_cleanup,
                         occlusion_enabled, occlusion_strength, occlusion_sensitivity,
                         # mouth protection params
                         mouth_protect_enabled=True, mouth_protect_strength=0.75,
                         mouth_open_threshold=0.28,
                         # auto face fix
                         auto_face_fix=True,
                         # auto face match params
                         auto_match_enabled=True, auto_match_threshold=0.20,
                         source_image_2=None, source_image_3=None,
                         # adaptive params
                         adaptive_enabled=False, adaptive_max_retries=1,
                         adaptive_confidence_threshold=0.30,
                         adaptive_color_match=False,
                         adaptive_swap_strength=0.0,
                         adaptive_restore_strength=0.0,
                         adaptive_sharpen=0.0,
                         adaptive_texture_blend=0.0):
        """
        Process each image individually BEFORE it gets saved.
        Supports both classic mode and new adaptive pipeline mode.
        """
        if not enabled:
            print("[ReActor V3] Skipped - checkbox is not enabled")
            return
        if source_image is None:
            print("[ReActor V3] Skipped - no source image provided")
            return
        if restore_model == 'None' and not adaptive_enabled and not auto_match_enabled:
            print("[ReActor V3] Skipped - no restore model selected and adaptive/auto-match disabled")
            return

        try:
            print("")
            print("[ReActor V3] ========================================")
            print("[ReActor V3]   POST-PROCESS IMAGE TRIGGERED")
            print("[ReActor V3] ========================================")
            total_start = time.time()
            
            # Log all settings
            print(f"[ReActor V3]   Auto Face Match: {auto_match_enabled}")
            if auto_match_enabled:
                print(f"[ReActor V3]   Match threshold: {auto_match_threshold}")
                has_extra_src = source_image_2 is not None or source_image_3 is not None
                print(f"[ReActor V3]   Additional sources: {has_extra_src}")
            else:
                print(f"[ReActor V3]   Source face index: {source_face_index}")
                print(f"[ReActor V3]   Target face index: {target_face_index}")
            print(f"[ReActor V3]   Restore model: {restore_model}")
            print(f"[ReActor V3]   Gender match: {gender_match}")
            print(f"[ReActor V3]   Auto resolution: {auto_resolution}")
            print(f"[ReActor V3]   Aggressive cleanup: {aggressive_cleanup}")
            print(f"[ReActor V3]   Occlusion: enabled={occlusion_enabled}, strength={occlusion_strength}, sensitivity={occlusion_sensitivity}")
            print(f"[ReActor V3]   Mouth protection: enabled={mouth_protect_enabled}, strength={mouth_protect_strength}, threshold={mouth_open_threshold}")
            print(f"[ReActor V3]   Auto face detail fix: {auto_face_fix}")
            print(f"[ReActor V3]   Adaptive pipeline: {adaptive_enabled}")
            if adaptive_enabled:
                print(f"[ReActor V3]   Adaptive max retries: {adaptive_max_retries}")
                print(f"[ReActor V3]   Adaptive confidence threshold: {adaptive_confidence_threshold}")
                print(f"[ReActor V3]   Adaptive color match: {adaptive_color_match}")
                print(f"[ReActor V3]   Adaptive swap strength: {adaptive_swap_strength}")
                print(f"[ReActor V3]   Adaptive restore strength: {adaptive_restore_strength}")
                print(f"[ReActor V3]   Adaptive sharpen: {adaptive_sharpen}")
                print(f"[ReActor V3]   Adaptive texture blend: {adaptive_texture_blend}")
            
            if torch.cuda.is_available():
                vram_alloc = torch.cuda.memory_allocated() / (1024**3)
                vram_res = torch.cuda.memory_reserved() / (1024**3)
                print(f"[ReActor V3]   VRAM at start - Allocated: {vram_alloc:.2f} GB, Reserved: {vram_res:.2f} GB")
            # ── Resolve paths & engine ──────────────────────────────────
            script_dir    = os.path.dirname(os.path.abspath(__file__))
            extension_dir = os.path.dirname(script_dir)
            extensions_dir= os.path.dirname(extension_dir)
            webui_dir     = os.path.dirname(extensions_dir)
            models_path   = os.path.join(webui_dir, 'models')

            engine = get_reactor_v3_engine(models_path)
            engine.set_cleanup_mode(aggressive_cleanup)
            engine.set_occlusion_handling(
                enabled=bool(occlusion_enabled),
                strength=float(occlusion_strength),
                sensitivity=float(occlusion_sensitivity),
            )
            engine.set_mouth_protection(
                enabled=bool(mouth_protect_enabled),
                strength=float(mouth_protect_strength),
                threshold=float(mouth_open_threshold),
            )
            engine.set_auto_face_fix(bool(auto_face_fix))

            source_cv2 = pil_to_cv2(source_image)
            target_cv2 = pil_to_cv2(pp.image)
            if source_cv2 is None or target_cv2 is None:
                print("[ReActor V3]   ERROR: Failed to convert images to CV2 format")
                return
            
            print(f"[ReActor V3]   Source CV2: {source_cv2.shape[1]}x{source_cv2.shape[0]} ({source_cv2.dtype})")
            print(f"[ReActor V3]   Target CV2: {target_cv2.shape[1]}x{target_cv2.shape[0]} ({target_cv2.dtype})")
            
            # Analyze source vs target color/brightness for mismatch warning
            src_mean = np.mean(source_cv2, axis=(0,1))
            tgt_mean = np.mean(target_cv2, axis=(0,1))
            src_brightness = float(np.mean(cv2.cvtColor(source_cv2, cv2.COLOR_BGR2GRAY)))
            tgt_brightness = float(np.mean(cv2.cvtColor(target_cv2, cv2.COLOR_BGR2GRAY)))
            print(f"[ReActor V3]   Source mean color (BGR): [{src_mean[0]:.1f}, {src_mean[1]:.1f}, {src_mean[2]:.1f}], brightness: {src_brightness:.1f}")
            print(f"[ReActor V3]   Target mean color (BGR): [{tgt_mean[0]:.1f}, {tgt_mean[1]:.1f}, {tgt_mean[2]:.1f}], brightness: {tgt_brightness:.1f}")
            brightness_diff = abs(src_brightness - tgt_brightness)
            if brightness_diff > 30:
                print(f"[ReActor V3]   \u26a0 BRIGHTNESS MISMATCH between source ({src_brightness:.0f}) and target ({tgt_brightness:.0f}): diff={brightness_diff:.0f}")
            color_diff = np.mean(np.abs(src_mean - tgt_mean))
            if color_diff > 25:
                print(f"[ReActor V3]   \u26a0 COLOR TONE MISMATCH between source and target: avg_diff={color_diff:.1f}")

            # ─────────────────────────────────────────────────────────────
            if auto_match_enabled:
                # ── AUTO FACE MATCH PIPELINE PATH ────────────────────────
                # Collect additional source images
                additional_sources = []
                for extra_src in [source_image_2, source_image_3]:
                    if extra_src is not None:
                        extra_cv2 = pil_to_cv2(extra_src)
                        if extra_cv2 is not None:
                            additional_sources.append(extra_cv2)
                
                effective_restore = restore_model if restore_model != 'None' else None
                
                print(f"[ReActor V3] Running Auto Face Match Pipeline...")
                print(f"[ReActor V3]   Primary source + {len(additional_sources)} additional source(s)")
                result_cv2, status = engine.process_auto_match(
                    source_img           = source_cv2,
                    target_img           = target_cv2,
                    restore_model        = effective_restore,
                    gender_match         = gender_match,
                    similarity_threshold = float(auto_match_threshold),
                    additional_sources   = additional_sources if additional_sources else None,
                )
                print(f"[ReActor V3] {status}")
            
            elif adaptive_enabled:
                # ── ADAPTIVE PIPELINE PATH ───────────────────────────────
                import reactor_v3_adaptive as _adp

                # Override global confidence threshold from UI slider
                _adp.CONFIDENCE_REVIEW_THRESHOLD = float(adaptive_confidence_threshold)
                _adp.MAX_RETRIES                 = int(adaptive_max_retries)

                pipeline = get_adaptive_pipeline(engine)

                # Build optional force_params from manual override sliders
                force_params = None
                if (adaptive_swap_strength > 0.0 or adaptive_restore_strength > 0.0
                        or adaptive_sharpen > 0.0 or adaptive_texture_blend > 0.0
                        or adaptive_color_match):
                    force_params = AdaptiveParams(
                        swap_strength         = float(adaptive_swap_strength) if adaptive_swap_strength > 0 else 1.0,
                        restore_strength      = float(adaptive_restore_strength) if adaptive_restore_strength > 0 else 0.85,
                        restore_model         = restore_model if restore_model != 'None' else 'auto',
                        color_match           = bool(adaptive_color_match),
                        color_match_strength  = 0.55,
                        sharpen_strength      = float(adaptive_sharpen)      if adaptive_sharpen > 0 else 0.30,
                        texture_preserve_blend= float(adaptive_texture_blend),
                        reason                = "manual_override",
                    )

                print("[ReActor V3] Running Adaptive Pipeline...")
                result_cv2, report = pipeline.run(
                    source_img        = source_cv2,
                    target_img        = target_cv2,
                    source_face_index = int(source_face_index),
                    target_face_index = int(target_face_index),
                    gender_match      = gender_match,
                    force_params      = force_params,
                )
                print(report.summary())

                if report.flagged_for_review:
                    # Don't replace image — leave original + print warning
                    print("[ReActor V3] ⚠ Image flagged for manual review — original kept.")
                    return

            else:
                # ── CLASSIC PIPELINE PATH (unchanged behaviour) ───────────
                if not engine.load_restorer(restore_model):
                    return

                print("[ReActor V3] Processing image (classic mode)...")
                result_cv2, status = engine.process(
                    source_img        = source_cv2,
                    target_img        = target_cv2,
                    source_face_index = int(source_face_index),
                    target_face_index = int(target_face_index),
                    restore_model     = restore_model,
                    gender_match      = gender_match,
                )
                print(f"[ReActor V3] {status}")

            # ── Replace image in-place ───────────────────────────────────
            result_pil = cv2_to_pil(result_cv2)
            if result_pil is not None:
                pp.image = result_pil
                print(f"[ReActor V3]   Output image replaced: {result_pil.size[0]}x{result_pil.size[1]}")
            
            total_elapsed = time.time() - total_start
            if torch.cuda.is_available():
                vram_alloc = torch.cuda.memory_allocated() / (1024**3)
                vram_res = torch.cuda.memory_reserved() / (1024**3)
                print(f"[ReActor V3]   VRAM at end - Allocated: {vram_alloc:.2f} GB, Reserved: {vram_res:.2f} GB")
            print(f"[ReActor V3]   Total postprocess time: {total_elapsed:.3f}s")
            print("[ReActor V3] ========================================")
            print("")

        except Exception as e:
            print(f"[ReActor V3] Error in postprocess_image: {e}")
            import traceback
            traceback.print_exc()


print("[ReActor V3] Script loaded successfully")
print("[ReActor V3] Will appear in img2img/txt2img tabs alongside other reactor extensions")
print("[ReActor V3] Place GPEN models in: extensions/sd-webui-reactor-v3/models/facerestore_models/")
