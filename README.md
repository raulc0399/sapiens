# Meta Sapiens Body Part Segmentation

This script performs body part segmentation on images using Meta's Sapiens models. It processes images to identify and segment different body parts, generating both visual and data outputs.

## Repository

For more information and updates, visit the [Meta Sapiens GitHub repository](https://github.com/facebookresearch/sapiens).

## Requirements

```bash
pip install -r requirements.txt
```

## Models

Place the Sapiens model files in `../models/sapiens/`. The script supports the following model variants:
- 0.3b
- 0.6b
- 1b
- 2b

Model files should follow the naming convention in the script, e.g., `sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2`

## Usage

Basic usage:
```bash
python seg.py /path/to/input/images --output_dir /path/to/output --model 1b
```

Arguments:
- `input_dir`: Directory containing input images (required)
- `--output_dir`: Directory to save results (default: 'output')
- `--model`: Model size to use (choices: '0.3b', '0.6b', '1b', '2b', default: '1b')

## Outputs

For each input image, the script generates three output files:

1. `*_segmentation.png`: Visualization of the segmentation overlaid on the original image
2. `*_mask.npy`: Raw numerical mask data as a NumPy array
3. `*_mask_viz.png`: Visualization of the segmentation mask with class labels

## Features

- Batch processing of multiple images
- CUDA support for GPU acceleration
- Progress feedback during processing
- Error handling for individual images
- Memory usage
- Detailed model information display
