"""
Setup script to create the required directory structure for YOLOv8 segmentation training
"""
import os

# Base data directory
BASE_DIR = os.path.join("..", "data")

# Directories to create
DIRECTORIES = [
    os.path.join(BASE_DIR, "images", "train"),
    os.path.join(BASE_DIR, "images", "val"),
    os.path.join(BASE_DIR, "images", "test"),
    os.path.join(BASE_DIR, "labels", "train"),
    os.path.join(BASE_DIR, "labels", "val"),
    os.path.join(BASE_DIR, "labels", "test"),
]

def setup_directories():
    """Create all necessary directories for the dataset"""
    print("Setting up directory structure...")
    print("=" * 60)
    
    for directory in DIRECTORIES:
        os.makedirs(directory, exist_ok=True)
        abs_path = os.path.abspath(directory)
        print(f"✓ Created: {abs_path}")
    
    print("=" * 60)
    print("\nDirectory structure created successfully!")
    print("\nNext steps:")
    print("1. Place your training images in: data/images/train/")
    print("2. Place your training labels in: data/labels/train/")
    print("3. Place your validation images in: data/images/val/")
    print("4. Place your validation labels in: data/labels/val/")
    print("5. Update data/dataset.yaml with your class names")
    print("6. Run: python train_seg.py")
    
    # Display directory tree
    print("\n" + "=" * 60)
    print("Directory Structure:")
    print("=" * 60)
    print("""
data/
├── dataset.yaml
├── images/
│   ├── train/      ← Place training images here
│   ├── val/        ← Place validation images here
│   └── test/       ← Place test images here (optional)
└── labels/
    ├── train/      ← Place training labels here
    ├── val/        ← Place validation labels here
    └── test/       ← Place test labels here (optional)
    """)

if __name__ == "__main__":
    setup_directories()
