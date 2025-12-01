import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from pathlib import Path

# Import from your existing project files
from plant_dataset import PlantVillageDataLoader
from cnn_model import LeafDiseaseResNet18
from train import Trainer  # Reuse your Trainer class!


def fine_tune_model():
    # ===== CONFIGURATION =====
    # Your specific run folder containing the best model
    PREV_RUN_FOLDER = r"C:\Users\Asus\PycharmProjects\LeafDisease-CNN\src\runs\LeafDisease_ResNet18_20251127_161904"
    CHECKPOINT_PATH = os.path.join(PREV_RUN_FOLDER, "checkpoints", "best_model.pth")

    # Data path (remains the same)
    DATA_DIR = r"C:\Users\Asus\PycharmProjects\LeafDisease-CNN\data\PlantVillage"

    # Fine-tuning Settings
    NEW_EPOCHS = 3  # Train for just 3 more epochs
    LOW_LEARNING_RATE = 1e-4  # Use a smaller learning rate for gentle improvement
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"🔍 Looking for model at: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print("❌ Error: Model file not found! Please verify the PREV_RUN_FOLDER path.")
        return

    # 1. Load Data
    print("\nLoading dataset...")
    data_loader = PlantVillageDataLoader(DATA_DIR)
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0  # For Windows
    )

    # 2. Initialize Model
    print("Initializing model architecture...")
    model = LeafDiseaseResNet18(num_classes=16, pretrained=False)

    # 3. Load Pre-trained Weights
    print("Loading your trained weights...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded successfully (Previous Val Accuracy: {checkpoint['val_acc'] * 100:.2f}%)")

    # 4. Setup Trainer with a new experiment name
    # This will create a NEW folder in /runs for the fine-tuned model
    trainer = Trainer(model, device=DEVICE, experiment_name='LeafDisease_FineTuned')

    # 5. Start Fine-Tuning
    print(f"\n🚀 Starting Fine-Tuning for {NEW_EPOCHS} epochs...")
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=NEW_EPOCHS,
        learning_rate=LOW_LEARNING_RATE,  # Using the lower learning rate
        weight_decay=1e-4
    )

    # 6. Test the newly Fine-Tuned Model
    print("\n🔬 Evaluating the fine-tuned model on the test set...")
    test_acc, test_prec, test_rec, test_f1 = trainer.test_epoch(test_loader, data_loader.class_names)

    # 7. Save History and Config
    trainer.save_history()

    class_mapping_path = trainer.run_dir / "class_mapping.json"
    with open(class_mapping_path, 'w') as f:
        json.dump(data_loader.idx_to_class, f, indent=2)

    print(f"\n✅ Fine-tuning Complete! New, improved model saved in: {trainer.run_dir}")


if __name__ == "__main__":
    fine_tune_model()
