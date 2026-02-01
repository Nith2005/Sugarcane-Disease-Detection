from ultralytics import YOLO
import os

# Configuration
MODEL_TYPE = "yolov8n-seg.pt"  # Options: yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg
DATA_YAML = "../data/dataset.yaml"  # Path to your dataset configuration
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16  # Adjust based on your GPU memory
PROJECT_NAME = "sugarcane_seg_training"
EXPERIMENT_NAME = "yolov8_seg_v1"

def train_segmentation_model():
    """
    Train YOLOv8 segmentation model for sugarcane disease and insect detection
    """
    print("=" * 60)
    print("YOLOv8 Segmentation Training")
    print("=" * 60)
    
    # Load a pretrained YOLOv8 segmentation model
    print(f"\nLoading model: {MODEL_TYPE}")
    model = YOLO(MODEL_TYPE)
    
    # Display model information
    print(f"\nModel loaded successfully!")
    print(f"Dataset config: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Train the model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        patience=50,  # Early stopping patience
        save=True,  # Save checkpoints
        save_period=10,  # Save checkpoint every 10 epochs
        cache=False,  # Cache images for faster training (set to True if you have enough RAM)
        device=0,  # GPU device (0 for first GPU, 'cpu' for CPU)
        workers=8,  # Number of worker threads
        pretrained=True,  # Use pretrained weights
        optimizer='auto',  # Optimizer (auto, SGD, Adam, AdamW)
        verbose=True,  # Verbose output
        seed=42,  # Random seed for reproducibility
        deterministic=True,  # Deterministic training
        single_cls=False,  # Train as single-class dataset
        rect=False,  # Rectangular training
        cos_lr=False,  # Use cosine learning rate scheduler
        close_mosaic=10,  # Disable mosaic augmentation for final epochs
        resume=False,  # Resume training from last checkpoint
        amp=True,  # Automatic Mixed Precision training
        fraction=1.0,  # Dataset fraction to train on
        profile=False,  # Profile ONNX and TensorRT speeds
        freeze=None,  # Freeze layers (list of layer indices or number of layers)
        # Data augmentation parameters
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation augmentation
        hsv_v=0.4,  # HSV-Value augmentation
        degrees=0.0,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,  # Scale augmentation
        shear=0.0,  # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.0,  # Vertical flip probability
        fliplr=0.5,  # Horizontal flip probability
        mosaic=1.0,  # Mosaic augmentation probability
        mixup=0.0,  # Mixup augmentation probability
        copy_paste=0.0,  # Copy-paste augmentation probability
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Display results
    print(f"\nResults saved to: {PROJECT_NAME}/{EXPERIMENT_NAME}")
    print(f"Best model saved at: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt")
    print(f"Last model saved at: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/last.pt")
    
    # Validate the model
    print("\n" + "=" * 60)
    print("Running validation...")
    print("=" * 60 + "\n")
    
    metrics = model.val()
    
    print("\nValidation Metrics:")
    print(f"mAP50: {metrics.seg.map50:.4f}")
    print(f"mAP50-95: {metrics.seg.map:.4f}")
    
    return results, metrics

if __name__ == "__main__":
    try:
        results, metrics = train_segmentation_model()
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        raise
