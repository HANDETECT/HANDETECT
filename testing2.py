import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define your dataset directories
TRAIN_DATA_DIR = r'C:\Users\User\Desktop\SpiralSense\data\train\Alzheimers Disease'
VAL_DATA_DIR = r'C:\Users\User\Desktop\SpiralSense\data\val'  # Update with actual path
TEST_DATA_DIR = r'C:\Users\User\Desktop\SpiralSense\data\test\Alzheimers Disease'  # Update with actual path

# Define any transformations you want to apply
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Debugging: Print directory paths
print(f"Training data directory: {TRAIN_DATA_DIR}")
print(f"Validation data directory: {VAL_DATA_DIR}")
print(f"Test data directory: {TEST_DATA_DIR}")

# Debugging: List files in directories
print("Training directory files:", os.listdir(TRAIN_DATA_DIR) if os.path.exists(TRAIN_DATA_DIR) else "Directory does not exist")
print("Validation directory files:", os.listdir(VAL_DATA_DIR) if os.path.exists(VAL_DATA_DIR) else "Directory does not exist")
print("Test directory files:", os.listdir(TEST_DATA_DIR) if os.path.exists(TEST_DATA_DIR) else "Directory does not exist")

# Load datasets
class CustomDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

# Create dataset instances
train_dataset = CustomDataset(TRAIN_DATA_DIR, transform=transform)
val_dataset = CustomDataset(VAL_DATA_DIR, transform=transform)
test_dataset = CustomDataset(TEST_DATA_DIR, transform=transform)

# Print the lengths of the datasets to confirm they are loaded correctly
print(f"Training dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")
