import os
from plant_dataset import PlantVillageDataLoader

# Use absolute path - CORRECT FOR YOUR SYSTEM
data_dir = r"C:\Users\Asus\PycharmProjects\LeafDisease-CNN\data\PlantVillage"

# Verify the path exists
if not os.path.exists(data_dir):
    print(f"❌ ERROR: Path does not exist: {data_dir}")
    print("Please check your data directory!")
else:
    print(f"✓ Path found: {data_dir}")

    data_loader = PlantVillageDataLoader(data_dir)
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(batch_size=32)

    # Print class info
    print(f"\nClasses: {data_loader.class_names}")
    print(f"Total classes: {len(data_loader.class_names)}")
