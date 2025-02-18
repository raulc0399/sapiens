import os
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from gradio.themes.utils import sizes
from torchvision import transforms
from PIL import Image
import tempfile
from classes_and_palettes import GOLIATH_PALETTE, GOLIATH_CLASSES

class Config:
    ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
    CHECKPOINTS_DIR = os.path.join(ASSETS_DIR, "checkpoints")
    CHECKPOINTS = {
        "0.3b": "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2",
        "0.6b": "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2",
        "1b": "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
    }

class ModelManager:
    @staticmethod
    def load_model(checkpoint_name: str):
        checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, Config.CHECKPOINTS[checkpoint_name])
        model = torch.jit.load(checkpoint_path)
        model.eval()
        model.to("cuda")
        return model

    @staticmethod
    @torch.inference_mode()
    def run_model(model, input_tensor, height, width):
        output = model(input_tensor)
        output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
        _, preds = torch.max(output, 1)
        return preds

class ImageProcessor:
    def __init__(self):
        self.transform_fn = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], std=[58.5/255, 57.0/255, 57.5/255]),
        ])

    def process_image(self, image: Image.Image, model_name: str):
        model = ModelManager.load_model(model_name)
        input_tensor = self.transform_fn(image).unsqueeze(0).to("cuda")
        
        preds = ModelManager.run_model(model, input_tensor, image.height, image.width)
        mask = preds.squeeze(0).cpu().numpy()

        # Visualize the segmentation
        blended_image = self.visualize_pred_with_overlay(image, mask)

        # Create downloadable .npy file
        npy_path = tempfile.mktemp(suffix='.npy')
        np.save(npy_path, mask)

        return blended_image, npy_path

    @staticmethod
    def visualize_pred_with_overlay(img, sem_seg, alpha=0.5):
        img_np = np.array(img.convert("RGB"))
        sem_seg = np.array(sem_seg)

        num_classes = len(GOLIATH_CLASSES)
        ids = np.unique(sem_seg)[::-1]
        legal_indices = ids < num_classes
        ids = ids[legal_indices]
        labels = np.array(ids, dtype=np.int64)

        colors = [GOLIATH_PALETTE[label] for label in labels]

        overlay = np.zeros((*sem_seg.shape, 3), dtype=np.uint8)

        for label, color in zip(labels, colors):
            overlay[sem_seg == label, :] = color

        blended = np.uint8(img_np * (1 - alpha) + overlay * alpha)
        return Image.fromarray(blended)

class GradioInterface:
    def __init__(self):
        self.image_processor = ImageProcessor()

    def create_interface(self):
        app_styles = """
        <style>
            /* Global Styles */
            body, #root {
                font-family: Helvetica, Arial, sans-serif;
                background-color: #1a1a1a;
                color: #fafafa;
            }
            /* Header Styles */
            .app-header {
                background: linear-gradient(45deg, #1a1a1a 0%, #333333 100%);
                padding: 24px;
                border-radius: 8px;
                margin-bottom: 24px;
                text-align: center;
            }
            .app-title {
                font-size: 48px;
                margin: 0;
                color: #fafafa;
            }
            .app-subtitle {
                font-size: 24px;
                margin: 8px 0 16px;
                color: #fafafa;
            }
            .app-description {
                font-size: 16px;
                line-height: 1.6;
                opacity: 0.8;
                margin-bottom: 24px;
            }
            /* Button Styles */
            .publication-links {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 16px;
            }
            .publication-link {
                display: inline-flex;
                align-items: center;
                padding: 8px 16px;
                background-color: #333;
                color: #fff !important;
                text-decoration: none !important;
                border-radius: 20px;
                font-size: 14px;
                transition: background-color 0.3s;
            }
            .publication-link:hover {
                background-color: #555;
            }
            .publication-link i {
                margin-right: 8px;
            }
            /* Content Styles */
            .content-container {
                background-color: #2a2a2a;
                border-radius: 8px;
                padding: 24px;
                margin-bottom: 24px;
            }
            /* Image Styles */
            /* Updated Image Styles */
            .image-preview img {
                max-width: 512px;
                max-height: 512px;  
                margin: 0 auto;
                border-radius: 4px;
                display: block;
                object-fit: contain;  
            }

            /* Control Styles */
            .control-panel {
                background-color: #333;
                padding: 16px;
                border-radius: 8px;
                margin-top: 16px;
            }
            /* Gradio Component Overrides */
            .gr-button {
                background-color: #4a4a4a;
                color: #fff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .gr-button:hover {
                background-color: #5a5a5a;
            }
            .gr-input, .gr-dropdown {
                background-color: #3a3a3a;
                color: #fff;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 8px;
            }
            .gr-form {
                background-color: transparent;
            }
            .gr-panel {
                border: none;
                background-color: transparent;
            }
            /* Override any conflicting styles from Bulma */
            .button.is-normal.is-rounded.is-dark {
                color: #fff !important;
                text-decoration: none !important;
            }
        </style>
        """

        header_html = f"""
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.3/css/bulma.min.css">
        <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.15.4/css/all.css">
        {app_styles}
        <div class="app-header">
            <h1 class="app-title">Sapiens:Body-Part Segmentation</h1>
            <h2 class="app-subtitle">ECCV 2024 (Oral)</h2>
            <p class="app-description">
                Meta presents Sapiens, foundation models for human tasks pretrained on 300 million human images. 
                This demo showcases the finetuned body-part segmentation model. <br>
            </p>
            <div class="publication-links">
                <a href="https://arxiv.org/abs/2408.12569" class="publication-link">
                    <i class="fas fa-file-pdf"></i>arXiv
                </a>
                <a href="https://github.com/facebookresearch/sapiens" class="publication-link">
                    <i class="fab fa-github"></i>Code
                </a>
                <a href="https://about.meta.com/realitylabs/codecavatars/sapiens/" class="publication-link">
                    <i class="fas fa-globe"></i>Meta
                </a>
                <a href="https://rawalkhirodkar.github.io/sapiens" class="publication-link">
                    <i class="fas fa-chart-bar"></i>Results
                </a>
            </div>
        </div>
        """

        js_func = """
        function refresh() {
            const url = new URL(window.location);
            if (url.searchParams.get('__theme') !== 'dark') {
                url.searchParams.set('__theme', 'dark');
                window.location.href = url.href;
            }
        }
        """

        def process_image(image, model_name):
            result, npy_path = self.image_processor.process_image(image, model_name)
            return result, npy_path

        with gr.Blocks(js=js_func, theme=gr.themes.Default()) as demo:
            gr.HTML(header_html)
            with gr.Row(elem_classes="content-container"):
                with gr.Column():
                    input_image = gr.Image(label="Input Image", type="pil", format="png", elem_classes="image-preview")
                    model_name = gr.Dropdown(
                        label="Model Size",
                        choices=list(Config.CHECKPOINTS.keys()),
                        value="1b",
                    )
                    example_model = gr.Examples(
                        inputs=input_image,
                        examples_per_page=14,
                        examples=[
                            os.path.join(Config.ASSETS_DIR, "images", img)
                            for img in os.listdir(os.path.join(Config.ASSETS_DIR, "images"))
                        ],
                    )
                with gr.Column():
                    result_image = gr.Image(label="Segmentation Result", type="pil", elem_classes="image-preview")
                    npy_output = gr.File(label="Segmentation (.npy)")
                    run_button = gr.Button("Run")
                    gr.Image(os.path.join(Config.ASSETS_DIR, "palette.jpg"), label="Class Palette", type="filepath", elem_classes="image-preview")

            run_button.click(
                fn=process_image,
                inputs=[input_image, model_name],
                outputs=[result_image, npy_output],
            )
            
        return demo

def main():
    # Configure CUDA if available
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    interface = GradioInterface()
    demo = interface.create_interface()
    demo.launch(share=False)

if __name__ == "__main__":
    main()