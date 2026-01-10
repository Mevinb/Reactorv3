"""
ReActor V3 - WebUI Forge Script Integration

This script integrates ReActor V3 into img2img and txt2img tabs as a post-processing
script that automatically enhances faces after SD generation using GPEN-512/1024.
"""

import os
import sys
import gradio as gr
import modules.scripts as scripts
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
            
            gr.Markdown("""
            **💡 Tips:**
            - Upload source face image above
            - Enable checkbox to process generated images
            - GPEN-512: Fast, good quality (full body shots)
            - GPEN-1024: Slower, maximum quality (portraits)
            - Auto-resolution recommended for best speed/quality balance
            - **Aggressive Cleanup**: Enable if you have limited VRAM (<12GB) for consistent speed
            
            **📁 Model Location:** `extensions/sd-webui-reactor-v3/models/facerestore_models/`
            """)
            
            # Refresh button handler
            refresh_button.click(
                fn=lambda: gr.Dropdown.update(choices=get_available_models()),
                inputs=[],
                outputs=[restore_model]
            )
        
        return [enabled, source_image, source_face_index, target_face_index, 
                restore_model, auto_resolution, aggressive_cleanup]
    
    def postprocess_image(self, p, pp, enabled, source_image, source_face_index,
                         target_face_index, restore_model, auto_resolution, aggressive_cleanup):
        """
        Process each image individually BEFORE it gets saved.
        This is called per-image and runs before saving, ensuring processed versions get saved.
        """
        if not enabled:
            return
        
        if source_image is None:
            return
        
        if restore_model == 'None':
            return
        
        try:
            # Get extension path and use shared WebUI models
            script_dir = os.path.dirname(os.path.abspath(__file__))
            extension_dir = os.path.dirname(script_dir)
            extensions_dir = os.path.dirname(extension_dir)
            webui_dir = os.path.dirname(extensions_dir)
            models_path = os.path.join(webui_dir, 'models')
            
            engine = get_reactor_v3_engine(models_path)
            
            # Set cleanup mode based on user preference
            engine.set_cleanup_mode(aggressive_cleanup)
            
            # Load restoration model (cached, so fast after first load)
            if not engine.load_restorer(restore_model):
                return
            
            # Convert source image to OpenCV
            source_cv2 = pil_to_cv2(source_image)
            if source_cv2 is None:
                return
            
            # Convert current image to OpenCV
            target_cv2 = pil_to_cv2(pp.image)
            if target_cv2 is None:
                return
            
            print(f"[ReActor V3] Processing image before save...")
            
            # Process with ReActor V3
            result_cv2, status = engine.process(
                source_img=source_cv2,
                target_img=target_cv2,
                source_face_index=int(source_face_index),
                target_face_index=int(target_face_index),
                restore_model=restore_model
            )
            
            # Replace the image in-place
            result_pil = cv2_to_pil(result_cv2)
            if result_pil is not None:
                pp.image = result_pil
                print(f"[ReActor V3] {status}")
            
            # Note: Cleanup is now handled automatically inside process() method
            # based on the aggressive_cleanup setting we configured above
            
        except Exception as e:
            print(f"[ReActor V3] Error in postprocess_image: {e}")
            import traceback
            traceback.print_exc()


print("[ReActor V3] Script loaded successfully")
print("[ReActor V3] Will appear in img2img/txt2img tabs alongside other reactor extensions")
print("[ReActor V3] Place GPEN models in: extensions/sd-webui-reactor-v3/models/facerestore_models/")
