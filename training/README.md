# YOLOv8 Segmentation Training Guide

## Overview
This guide will help you train a YOLOv8 segmentation model for sugarcane disease and insect detection.

## Prerequisites
- Python 3.8+
- Ultralytics YOLOv8 installed (`pip install ultralytics`)
- GPU with CUDA support (recommended)
- Labeled segmentation dataset in YOLO format

## Dataset Structure

Your dataset should be organized as follows:

```
data/
├── dataset.yaml          # Dataset configuration file
├── images/
│   ├── train/           # Training images
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/             # Validation images
│   │   ├── image1.jpg
│   │   └── ...
│   └── test/            # Test images (optional)
│       └── ...
└── labels/
    ├── train/           # Training labels (segmentation masks)
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    ├── val/             # Validation labels
    │   ├── image1.txt
    │   └── ...
    └── test/            # Test labels (optional)
        └── ...
```

## Label Format

YOLOv8 segmentation uses a specific format for labels. Each `.txt` file should contain:

```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ... <xn> <yn>
```

Where:
- `class_id`: Integer representing the class (0-indexed)
- `x1, y1, x2, y2, ...`: Normalized polygon coordinates (0-1 range)

Example:
```
0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4
1 0.5 0.5 0.7 0.5 0.7 0.7 0.5 0.7
```

## Configuration

### 1. Update `data/dataset.yaml`

Edit the dataset configuration file to match your classes:

```yaml
path: D:/VIT Git/FINAL/Sugarcane-Disease-Insect-Detection/data
train: images/train
val: images/val
test: images/test

nc: 3  # Number of classes

names:
  0: healthy
  1: disease
  2: insect
```

### 2. Update `training/train_seg.py`

Adjust training parameters at the top of the file:

```python
MODEL_TYPE = "yolov8n-seg.pt"  # Model size (n, s, m, l, x)
EPOCHS = 100                    # Number of training epochs
IMG_SIZE = 640                  # Input image size
BATCH_SIZE = 16                 # Batch size (adjust for GPU memory)
```

**Model sizes:**
- `yolov8n-seg.pt`: Nano (fastest, least accurate)
- `yolov8s-seg.pt`: Small
- `yolov8m-seg.pt`: Medium
- `yolov8l-seg.pt`: Large
- `yolov8x-seg.pt`: Extra Large (slowest, most accurate)

## Training

### Run Training

```bash
cd training
python train_seg.py
```

### Monitor Training

The script will:
1. Download the pretrained YOLOv8 segmentation model (first run only)
2. Train on your dataset
3. Save checkpoints every 10 epochs
4. Validate the model after training
5. Save results to `sugarcane_seg_training/yolov8_seg_v1/`

### Training Output

Results will be saved in:
```
training/sugarcane_seg_training/yolov8_seg_v1/
├── weights/
│   ├── best.pt          # Best model checkpoint
│   └── last.pt          # Last epoch checkpoint
├── results.png          # Training metrics plot
├── confusion_matrix.png # Confusion matrix
├── val_batch0_pred.jpg  # Validation predictions
└── ...
```

## Evaluation

After training, you can evaluate the model:

```python
from ultralytics import YOLO

model = YOLO('sugarcane_seg_training/yolov8_seg_v1/weights/best.pt')
metrics = model.val()

print(f"mAP50: {metrics.seg.map50}")
print(f"mAP50-95: {metrics.seg.map}")
```

## Inference

Use the trained model for predictions:

```python
from ultralytics import YOLO

model = YOLO('sugarcane_seg_training/yolov8_seg_v1/weights/best.pt')
results = model('path/to/image.jpg')

# Process results
for result in results:
    masks = result.masks  # Segmentation masks
    boxes = result.boxes  # Bounding boxes
    result.show()  # Display results
    result.save('output.jpg')  # Save results
```

## Tips for Better Results

1. **Data Quality**: Ensure high-quality annotations with accurate segmentation masks
2. **Data Augmentation**: The script includes augmentation parameters you can tune
3. **Batch Size**: Increase if you have more GPU memory (16, 32, 64)
4. **Image Size**: Try 640, 800, or 1024 (larger = more accurate but slower)
5. **Epochs**: Train for more epochs if the model hasn't converged
6. **Early Stopping**: The script uses patience=50 to stop if no improvement
7. **Learning Rate**: The optimizer is set to 'auto' but you can specify SGD/Adam/AdamW

## Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE` (try 8, 4, or 2)
- Reduce `IMG_SIZE` (try 416 or 512)
- Use a smaller model (yolov8n-seg instead of yolov8l-seg)

### Poor Performance
- Check label quality and format
- Increase training epochs
- Try different augmentation parameters
- Use a larger model
- Collect more training data

### Dataset Not Found
- Verify paths in `dataset.yaml`
- Ensure images and labels are in correct directories
- Check file extensions (.jpg, .png, .txt)

## Next Steps

After training:
1. Copy the best model to `models/` directory
2. Update inference scripts to use the new model
3. Test on real-world images
4. Fine-tune if needed

## Resources

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLOv8 Segmentation](https://docs.ultralytics.com/tasks/segment/)
- [Training Tips](https://docs.ultralytics.com/guides/model-training-tips/)
