import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from classes_and_palettes import GOLIATH_PALETTE, GOLIATH_CLASSES

# run using:
# python seg.py /path/to/input/images --output_dir /path/to/output --model 1b

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'sapiens')
CHECKPOINTS = {
    "0.3b": "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2",
    "0.6b": "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2",
    "1b": "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
    "2b": "sapiens_2b_goliath_best_goliath_mIoU_8131_epoch_200_torchscript.pt2",
}

def load_model(checkpoint_name: str):
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, CHECKPOINTS[checkpoint_name])
    print(f"\nModel Information:")
    print(f"Checkpoint name: {checkpoint_name}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Print initial CUDA memory state
    if torch.cuda.is_available():
        print(f"\nCUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Initial CUDA memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Reserved/Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
    
    # Load model and print memory after loading
    print("\nLoading model...")
    model = torch.jit.load(checkpoint_path)
    print(f"Model type: {type(model)}")
    
    # Move to CUDA and print memory again
    print("\nMoving model to CUDA...")
    model.eval()
    model.to("cuda")
    
    if torch.cuda.is_available():
        print(f"\nCUDA memory after moving to GPU:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Reserved/Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
    
    # Get model parameters and structure
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,}")
    
    # Print model structure
    print("\nModel Structure:")
    print(model.graph)
    
    return model

@torch.inference_mode()
def run_model(model, input_tensor, height, width):
    output = model(input_tensor)
    output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
    _, preds = torch.max(output, 1)
    return preds

def process_image(image_path: str, model: str, output_dir: str):
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    transform_fn = transforms.Compose([
        transforms.Resize((1024, 768)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], std=[58.5/255, 57.0/255, 57.5/255]),
    ])
    
    # Process image
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
    
    # Create visualization of the mask
    mask_viz_path = os.path.join(output_dir, f"{base_name}_mask_viz.png")
    visualize_mask(mask, mask_viz_path)
    
    return vis_path, mask_path, mask_viz_path

def visualize_mask(mask, save_path):
    """Visualize the segmentation mask with class labels."""
    plt.figure(figsize=(15, 10))
    
    # Create main image
    plt.subplot(1, 2, 1)
    im = plt.imshow(mask)
    plt.title('Segmentation Mask')
    
    # Create custom colorbar with class labels
    ax2 = plt.subplot(1, 2, 2)
    unique_classes = np.unique(mask)
    colors = [im.cmap(im.norm(value)) for value in unique_classes]
    patches = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in colors]
    
    # Get class names for the legend
    class_names = [GOLIATH_CLASSES[idx] if idx < len(GOLIATH_CLASSES) else f"Unknown ({idx})" 
                  for idx in unique_classes]
    
    # Create legend
    ax2.legend(patches, class_names, loc='center', frameon=False)
    ax2.axis('off')
    plt.title('Class Labels')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

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

    # Load model
    model = load_model(args.model)

    # Process all images in the input directory
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args.input_dir, filename)
            print(f"Processing {filename}...")
            try:
                vis_path, mask_path, mask_viz_path = process_image(image_path, model, args.output_dir)
                print(f"Saved results to:")
                print(f"  - Overlay: {vis_path}")
                print(f"  - Raw mask: {mask_path}")
                print(f"  - Mask visualization: {mask_viz_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()
