import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from classes_and_palettes import GOLIATH_PALETTE, GOLIATH_CLASSES

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'checkpoints')
CHECKPOINTS = {
    "0.3b": "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2",
    "0.6b": "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2",
    "1b": "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
    "2b": "sapiens_2b_goliath_best_goliath_mIoU_8131_epoch_200_torchscript.pt2",
}

def load_model(checkpoint_name: str):
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, CHECKPOINTS[checkpoint_name])
    model = torch.jit.load(checkpoint_path)
    model.eval()
    model.to("cuda")
    return model

@torch.inference_mode()
def run_model(model, input_tensor, height, width):
    output = model(input_tensor)
    output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
    _, preds = torch.max(output, 1)
    return preds

def process_image(image_path: str, model_name: str, output_dir: str):
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    transform_fn = transforms.Compose([
        transforms.Resize((1024, 768)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], std=[58.5/255, 57.0/255, 57.5/255]),
    ])
    
    # Process image
    model = load_model(model_name)
    input_tensor = transform_fn(image).unsqueeze(0).to("cuda")
    preds = run_model(model, input_tensor, image.height, image.width)
    mask = preds.squeeze(0).cpu().numpy()

    # Create output filenames
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(output_dir, f"{base_name}_segmentation.png")
    mask_path = os.path.join(output_dir, f"{base_name}_mask.npy")

    # Save results
    blended_image = visualize_pred_with_overlay(image, mask)
    blended_image.save(vis_path)
    np.save(mask_path, mask)
    
    return vis_path, mask_path

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

def main():
    parser = argparse.ArgumentParser(description='Process images for body-part segmentation')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('--output_dir', default='output', help='Directory to save results')
    parser.add_argument('--model', default='1b', choices=['0.3b', '0.6b', '1b', '2b'], help='Model size to use')
    args = parser.parse_args()

    # Configure CUDA if available
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all images in the input directory
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args.input_dir, filename)
            print(f"Processing {filename}...")
            try:
                vis_path, mask_path = process_image(image_path, args.model, args.output_dir)
                print(f"Saved results to {vis_path} and {mask_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()
