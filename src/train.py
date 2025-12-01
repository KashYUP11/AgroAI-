import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from plant_dataset import PlantVillageDataLoader
from cnn_model import LeafDiseaseResNet18


class Trainer:
    """
    Trainer class for LeafDisease CNN model.
    Handles training, validation, and testing loops.
    """

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu',
                 experiment_name='LeafDisease_v1'):
        """
        Args:
            model (nn.Module): CNN model to train
            device (str): Device to use ('cuda' or 'cpu')
            experiment_name (str): Name of the experiment for logging
        """
        self.model = model.to(device)
        self.device = device
        self.experiment_name = experiment_name

        # Create run directory
        self.run_dir = Path(f"runs/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # TensorBoard logger
        self.writer = SummaryWriter(str(self.run_dir / "logs"))

        # Metrics tracking
        self.best_val_acc = 0
        self.best_epoch = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }

        print(f"Training setup created. Checkpoints saved to: {self.checkpoint_dir}")
        print(f"Device: {self.device}")

    def train_epoch(self, train_loader, optimizer, criterion):
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        avg_acc = accuracy_score(all_labels, all_preds)

        return avg_loss, avg_acc

    def validate_epoch(self, val_loader, criterion):
        """
        Validate on validation set.
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(val_loader)
        avg_acc = accuracy_score(all_labels, all_preds)
        avg_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        return avg_loss, avg_acc, avg_f1

    def test_epoch(self, test_loader, class_names):
        """
        Test on test set and generate detailed metrics.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        progress_bar = tqdm(test_loader, desc="Testing", leave=False)

        with torch.no_grad():
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        test_acc = accuracy_score(all_labels, all_preds)
        test_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        test_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        print(f"\n{'=' * 50}")
        print(f"TEST SET RESULTS")
        print(f"{'=' * 50}")
        print(f"Accuracy:  {test_acc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall:    {test_recall:.4f}")
        print(f"F1-Score:  {test_f1:.4f}")

        # Per-class metrics
        print(f"\n{'Class':<40} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 76)

        for i, class_name in enumerate(class_names):
            class_mask = np.array(all_labels) == i
            if class_mask.sum() > 0:
                class_preds = np.array(all_preds)[class_mask]
                class_acc = (class_preds == i).sum() / class_mask.sum()
                precision = precision_score(all_labels, all_preds, labels=[i], average='weighted', zero_division=0)
                recall = recall_score(all_labels, all_preds, labels=[i], average='weighted', zero_division=0)
                f1 = f1_score(all_labels, all_preds, labels=[i], average='weighted', zero_division=0)
                print(f"{class_name:<40} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")

        return test_acc, test_precision, test_recall, test_f1

    def save_checkpoint(self, epoch, optimizer, val_acc, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")

    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=1e-3, weight_decay=1e-4):
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs (int): Number of training epochs
            learning_rate (float): Initial learning rate
            weight_decay (float): L2 regularization weight
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        print(f"\n{'=' * 60}")
        print(f"Starting Training: {self.experiment_name}")
        print(f"{'=' * 60}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Weight Decay: {weight_decay}")
        print(f"Device: {self.device}")
        print(f"{'=' * 60}\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)

            # Validate
            val_loss, val_acc, val_f1 = self.validate_epoch(val_loader, criterion)

            # Log metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('F1/val', val_f1, epoch)
            self.writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

            # Print epoch results
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | F1: {val_f1:.4f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(epoch, optimizer, val_acc, is_best=True)
            else:
                self.save_checkpoint(epoch, optimizer, val_acc)

            # Learning rate scheduling
            scheduler.step(val_acc)

        print(f"\n{'=' * 60}")
        print(f"Training Complete!")
        print(f"Best Val Accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch + 1})")
        print(f"{'=' * 60}")

        self.writer.close()

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.run_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to: {history_path}")


def main():
    """Main training script."""

    # ===== CONFIGURATION - UPDATE THE PATH HERE =====
    CONFIG = {
        'data_dir': r"C:\Users\Asus\PycharmProjects\LeafDisease-CNN\data\PlantVillage",  # ← YOUR ABSOLUTE PATH
        'batch_size': 32,
        'num_epochs': 3,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_classes': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"Config: {CONFIG}")

    # Verify path exists
    if not os.path.exists(CONFIG['data_dir']):
        print(f"\n❌ ERROR: Data directory not found!")
        print(f"Expected path: {CONFIG['data_dir']}")
        print("Please check that your PlantVillage dataset is in the correct location.")
        return

    print(f"✓ Data directory found: {CONFIG['data_dir']}")

    # Load data
    print("\nLoading dataset...")
    data_loader = PlantVillageDataLoader(CONFIG['data_dir'])
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        batch_size=CONFIG['batch_size'],
        num_workers=0 if os.name == 'nt' else 2  # 0 for Windows, 2 for Linux
    )

    # Create model
    print("Creating model...")
    model = LeafDiseaseResNet18(num_classes=CONFIG['num_classes'], pretrained=True, dropout_rate=0.5)

    # Create trainer
    trainer = Trainer(model, device=CONFIG['device'], experiment_name='LeafDisease_ResNet18')

    # Train
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Test
    print("\nEvaluating on test set...")
    test_acc, test_prec, test_rec, test_f1 = trainer.test_epoch(test_loader, data_loader.class_names)

    # Save history
    trainer.save_history()

    # Save config
    config_path = trainer.run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"✓ Config saved to: {config_path}")

    # Save class mapping for use in webapp
    class_mapping_path = trainer.run_dir / "class_mapping.json"
    with open(class_mapping_path, 'w') as f:
        json.dump(data_loader.idx_to_class, f, indent=2)
    print(f"✓ Class mapping saved to: {class_mapping_path}")

    print(f"\n✓ All results saved to: {trainer.run_dir}")


if __name__ == "__main__":
    main()
