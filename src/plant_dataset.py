import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import glob


class PlantVillageDataset(Dataset):
    """
    Custom PyTorch Dataset for PlantVillage leaf disease classification.
    Loads images from class folders and applies transformations.
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of image file paths
            labels (list): List of corresponding class labels (integers)
            transform (callable, optional): Optional image transformations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns a single image and its label.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image on error
            image = Image.new('RGB', (256, 256), color=(0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


class PlantVillageDataLoader:
    """
    Utility class to load PlantVillage dataset and create PyTorch DataLoaders.
    """

    def __init__(self, data_dir, train_split=0.8, val_split=0.1, test_split=0.1, seed=42):
        """
        Args:
            data_dir (str): Path to PlantVillage dataset directory
            train_split (float): Proportion of data for training (default: 0.8)
            val_split (float): Proportion of data for validation (default: 0.1)
            test_split (float): Proportion of data for testing (default: 0.1)
            seed (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.class_names = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.image_paths = []
        self.labels = []

    def load_data(self):
        """
        Loads all images and their class labels from the dataset directory.
        Folder names become class labels.
        """
        class_idx = 0

        # Iterate through each class folder
        for class_folder in sorted(os.listdir(self.data_dir)):
            class_path = os.path.join(self.data_dir, class_folder)

            # Skip non-directory items
            if not os.path.isdir(class_path):
                continue

            self.class_names.append(class_folder)
            self.class_to_idx[class_folder] = class_idx
            self.idx_to_class[class_idx] = class_folder

            # Load all images from this class
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            for ext in image_extensions:
                image_files = glob.glob(os.path.join(class_path, ext))
                for img_file in image_files:
                    self.image_paths.append(img_file)
                    self.labels.append(class_idx)

            class_idx += 1

        print(f"Loaded {len(self.image_paths)} images from {len(self.class_names)} classes")
        print(f"Classes: {self.class_names}")

    def get_transforms(self):
        """
        Define image transformation pipelines for train and val/test.
        """
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]  # ImageNet std
            )
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return train_transform, val_test_transform

    def split_data(self):
        """
        Splits data into train, validation, and test sets using stratified split.
        Returns indices for each split.
        """
        # First split: separate test set
        train_val_idx, test_idx = train_test_split(
            range(len(self.labels)),
            test_size=self.test_split,
            random_state=self.seed,
            stratify=self.labels
        )

        # Second split: separate validation from training
        train_labels = [self.labels[i] for i in train_val_idx]
        val_size = self.val_split / (self.train_split + self.val_split)

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=self.seed,
            stratify=train_labels
        )

        return train_idx, val_idx, test_idx

    def create_dataloaders(self, batch_size=32, num_workers=0):
        """
        Creates and returns train, validation, and test DataLoaders.

        Args:
            batch_size (int): Batch size for DataLoaders
            num_workers (int): Number of worker threads for data loading

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        self.load_data()
        train_transform, val_test_transform = self.get_transforms()

        # Split data
        train_idx, val_idx, test_idx = self.split_data()

        # Create datasets
        train_paths = [self.image_paths[i] for i in train_idx]
        train_labels = [self.labels[i] for i in train_idx]
        train_dataset = PlantVillageDataset(train_paths, train_labels, train_transform)

        val_paths = [self.image_paths[i] for i in val_idx]
        val_labels = [self.labels[i] for i in val_idx]
        val_dataset = PlantVillageDataset(val_paths, val_labels, val_test_transform)

        test_paths = [self.image_paths[i] for i in test_idx]
        test_labels = [self.labels[i] for i in test_idx]
        test_dataset = PlantVillageDataset(test_paths, test_labels, val_test_transform)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"\nData Split:")
        print(f"Train: {len(train_dataset)} images")
        print(f"Val: {len(val_dataset)} images")
        print(f"Test: {len(test_dataset)} images")

        return train_loader, val_loader, test_loader


# Example usage
# Example usage
if __name__ == "__main__":
    # ===== ABSOLUTE PATH - UPDATE THIS =====
    data_dir = r"C:\Users\Asus\PycharmProjects\LeafDisease-CNN\data\PlantVillage"

    # Verify path exists
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: Path does not exist: {data_dir}")
        print("Please check your data directory!")
    else:
        print(f"✓ Path found: {data_dir}\n")

        data_loader = PlantVillageDataLoader(data_dir)
        train_loader, val_loader, test_loader = data_loader.create_dataloaders(batch_size=32)

        # Test loading a batch
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Labels: {labels}")
            break

