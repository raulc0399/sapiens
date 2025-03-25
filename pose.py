import os
import argparse
from typing import List
import numpy as np
import torch
import json
from torchvision import transforms
from PIL import Image
import cv2
from classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_WHOLEBODY_KPTS_COLORS,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    GOLIATH_KEYPOINTS
)

import os

from detector_utils import (
            adapt_mmdet_pipeline,
            init_detector,
            process_images_detector,
        )

class Config:
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '../models/sapiens')
    CHECKPOINTS = {
        "0.3b": "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2",
        "0.6b": "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2",
        "1b": "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2",
    }
    DETECTION_CHECKPOINT = os.path.join(MODELS_DIR, 'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth')
    DETECTION_CONFIG = os.path.join(MODELS_DIR, 'rtmdet_m_640-8xb32_coco-person_no_nms.py')

class ModelManager:
    @staticmethod
    def load_model(checkpoint_name: str):
        if checkpoint_name is None:
            return None
        checkpoint_path = os.path.join(Config.MODELS_DIR, checkpoint_name)
        model = torch.jit.load(checkpoint_path)
        model.eval()
        model.to("cuda")
        return model

    @staticmethod
    @torch.inference_mode()
    def run_model(model, input_tensor):
        return model(input_tensor)

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], 
                                 std=[58.5/255, 57.0/255, 57.5/255])
        ])
        self.detector = init_detector(
            Config.DETECTION_CONFIG, Config.DETECTION_CHECKPOINT, device='cpu'
        )
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

    def detect_persons(self, image: Image.Image):
        # Convert PIL Image to tensor
        image = np.array(image)
        image = np.expand_dims(image, axis=0)

        # Perform person detection
        bboxes_batch = process_images_detector(
            image, 
            self.detector
        )
        bboxes = self.get_person_bboxes(bboxes_batch[0])  # Get bboxes for the first (and only) image
        
        return bboxes
    
    def get_person_bboxes(self, bboxes_batch, score_thr=0.3):
        person_bboxes = []
        for bbox in bboxes_batch:
            if len(bbox) == 5:  # [x1, y1, x2, y2, score]
                if bbox[4] > score_thr:
                    person_bboxes.append(bbox)
            elif len(bbox) == 4:  # [x1, y1, x2, y2]
                person_bboxes.append(bbox + [1.0])  # Add a default score of 1.0
        return person_bboxes

    @torch.inference_mode()
    def estimate_pose(self, image: Image.Image, bboxes: List[List[float]], model_name: str, kpt_threshold: float):
        pose_model = ModelManager.load_model(Config.CHECKPOINTS[model_name])
        
        result_image = image.copy()
        all_keypoints = []  # List to store keypoints for all persons

        for bbox in bboxes:
            cropped_img = self.crop_image(result_image, bbox)
            input_tensor = self.transform(cropped_img).unsqueeze(0).to("cuda")
            heatmaps = ModelManager.run_model(pose_model, input_tensor)
            keypoints = self.heatmaps_to_keypoints(heatmaps[0].cpu().numpy(), bbox)
            all_keypoints.append(keypoints)  # Collect keypoints
            result_image = self.draw_keypoints(result_image, keypoints, bbox, kpt_threshold)
        
        return result_image, all_keypoints

    def process_image(self, image: Image.Image, model_name: str, kpt_threshold: str):
        bboxes = self.detect_persons(image)
        result_image, keypoints = self.estimate_pose(image, bboxes, model_name, float(kpt_threshold))
        return result_image, keypoints

    def crop_image(self, image, bbox):
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
        elif len(bbox) >= 5:
            x1, y1, x2, y2, _ = map(int, bbox[:5])
        else:
            raise ValueError(f"Unexpected bbox format: {bbox}")
        
        crop = image.crop((x1, y1, x2, y2))
        return crop

    @staticmethod
    def heatmaps_to_keypoints(heatmaps, bbox):
        num_joints = heatmaps.shape[0]  # Should be 308
        keypoints = {}
        x1, y1, x2, y2 = map(int, bbox[:4])
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        for i, name in enumerate(GOLIATH_KEYPOINTS):
            if i < num_joints:
                heatmap = heatmaps[i]
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                conf = heatmap[y, x]
                # Convert coordinates to image frame
                x_image = x * bbox_width / 192 + x1
                y_image = y * bbox_height / 256 + y1
                keypoints[name] = (float(x_image), float(y_image), float(conf))
        return keypoints

    @staticmethod
    def draw_keypoints(image, keypoints, bbox, kpt_threshold):
        image = np.array(image)

        # Handle both 4 and 5-element bounding boxes
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
        elif len(bbox) >= 5:
            x1, y1, x2, y2, _ = map(int, bbox[:5])
        else:
            raise ValueError(f"Unexpected bbox format: {bbox}")
                
        # Calculate adaptive radius and thickness based on bounding box size
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_size = np.sqrt(bbox_width * bbox_height)
        
        radius = max(1, int(bbox_size * 0.006))  # minimum 1 pixel
        thickness = max(1, int(bbox_size * 0.006))  # minimum 1 pixel
        bbox_thickness = max(1, thickness//4)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), bbox_thickness)
        
        # Draw keypoints for arms and hands
        arm_hand_keypoints = {
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist"
        }
        for i, (name, (x, y, conf)) in enumerate(keypoints.items()):
            if name in arm_hand_keypoints and conf > kpt_threshold and i < len(GOLIATH_KPTS_COLORS):
                x_coord = int(x)
                y_coord = int(y)
                color = GOLIATH_KPTS_COLORS[i]
                cv2.circle(image, (x_coord, y_coord), radius, color, -1)

        # Draw skeleton
        for _, link_info in GOLIATH_SKELETON_INFO.items():
            pt1_name, pt2_name = link_info['link']
            color = link_info['color']
            
            if pt1_name in keypoints and pt2_name in keypoints:
                pt1 = keypoints[pt1_name]
                pt2 = keypoints[pt2_name]
                if pt1[2] > kpt_threshold and pt2[2] > kpt_threshold:
                    x1_coord = int(pt1[0])
                    y1_coord = int(pt1[1])
                    x2_coord = int(pt2[0])
                    y2_coord = int(pt2[1])
                    cv2.line(image, (x1_coord, y1_coord), (x2_coord, y2_coord), color, thickness=thickness)

        return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser(description='Process images for human pose estimation')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('--output_dir', default='output', help='Directory to save results')
    parser.add_argument('--model', default='1b', choices=['0.3b', '0.6b', '1b'], help='Model size to use')
    parser.add_argument('--kpt_threshold', default='0.3', help='Minimum keypoint confidence threshold')
    args = parser.parse_args()

    # Configure CUDA if available
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize image processor
    image_processor = ImageProcessor()

    # Process all images in the input directory
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args.input_dir, filename)
            print(f"Processing {filename}...")
            try:
                # Load image
                image = Image.open(image_path).convert("RGB")
                
                # Process image
                result_image, keypoints = image_processor.process_image(image, args.model, args.kpt_threshold)
                
                # Create output filenames
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                result_path = os.path.join(args.output_dir, f"{base_name}_pose.png")
                json_path = os.path.join(args.output_dir, f"{base_name}_keypoints.json")
                
                # Save results
                result_image.save(result_path)
                with open(json_path, 'w') as f:
                    json.dump(keypoints, f, indent=2)
                
                print(f"Saved results to:")
                print(f"  - Visualization: {result_path}")
                print(f"  - Keypoints: {json_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()
