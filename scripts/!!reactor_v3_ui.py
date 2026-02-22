"""
ReActor V3 - WebUI Forge Script Integration

This script integrates ReActor V3 into img2img and txt2img tabs as a post-processing
script that automatically enhances faces after SD generation using GPEN-512/1024.
"""

import os
import sys
import gradio as gr
import modules.scripts as scripts
from typing import Optional
from modules import images
from modules.processing import Processed
from modules.shared import opts, state
from PIL import Image
import numpy as np
import cv2

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
            - **Smart Match**: Automatically swaps only matching gender (e.g., female source → female target)
            - **Gender Filter**: Use M/F to swap only male or female faces regardless of source
            - **Aggressive Cleanup**: Enable if you have limited VRAM (<12GB) for consistent speed
            
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
            return
        if source_image is None:
            return
        if restore_model == 'None' and not adaptive_enabled:
            return

        try:
            # ── Resolve paths & engine ──────────────────────────────────
            script_dir    = os.path.dirname(os.path.abspath(__file__))
            extension_dir = os.path.dirname(script_dir)
            extensions_dir= os.path.dirname(extension_dir)
            webui_dir     = os.path.dirname(extensions_dir)
            models_path   = os.path.join(webui_dir, 'models')

            engine = get_reactor_v3_engine(models_path)
            engine.set_cleanup_mode(aggressive_cleanup)

            source_cv2 = pil_to_cv2(source_image)
            target_cv2 = pil_to_cv2(pp.image)
            if source_cv2 is None or target_cv2 is None:
                return

            # ─────────────────────────────────────────────────────────────
            if adaptive_enabled:
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

        except Exception as e:
            print(f"[ReActor V3] Error in postprocess_image: {e}")
            import traceback
            traceback.print_exc()


print("[ReActor V3] Script loaded successfully")
print("[ReActor V3] Will appear in img2img/txt2img tabs alongside other reactor extensions")
print("[ReActor V3] Place GPEN models in: extensions/sd-webui-reactor-v3/models/facerestore_models/")
