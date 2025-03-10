import os
import torch
import sys
import torch.optim as optim
import torch.nn as nn
from Industrio_Data_Loader import iDataLoader
from Industrio_CNN import iCNN
from torch.utils.data import DataLoader


def PromptTrain():

    message = "🧠 LET'S TRAIN INDUSTRIO 🧠"
    border = "*" * 50
    padding = (50 - len(message) - 2) // 2  # Calculate space padding

    print("\n" + border)
    print("*" + " " * padding + message + " " * (50 - len(message) - padding - 2) + "*")
    print(border + "\n")

    train_path = ""
    val_path = ""
    checkpoint_path = ""
    epochs = 50
    batch_size = 16
    cuda = False

    while True:
        train_path = input("Please enter the full path for the training folder: ").strip()
        if not os.path.exists(train_path):  # Check if the path is valid
            print("⚠️ Invalid path. Please enter a valid directory or file path. ⚠️")
        elif not os.path.exists(os.path.join(train_path, "ok")) or not os.path.exists(os.path.join(train_path, "nok")):
            print("💀 Train folder does not contain an \"ok\" and a \"nok\" folder. Please make sure these folders are"
                  "set up correctly before attempting to train Industrio! Press any key to exit. 💀")
            input()
            sys.exit()
        else:
            break

    while True:
        val_path = input("Please enter the full path for the validation folder: ").strip()
        if not os.path.exists(val_path):  # Check if the path is valid
            print("⚠️ Invalid path. Please enter a valid directory or file path. ⚠️")
        elif not os.path.exists(os.path.join(val_path, "ok")) or not os.path.exists(os.path.join(val_path, "nok")):
            print("💀 Validation folder does not contain an \"ok\" and a \"nok\" folder. Please make sure these "
                  "folders are set up correctly before attempting to train Industrio! Press any key to exit. 💀")
            input()
            sys.exit()
        else:
            break

    while True:
        checkpoint_path = input("Please enter the full path for the output model: ").strip()
        if not os.path.exists(checkpoint_path):  # Check if the path is valid
            print("⚠️ Invalid path. Please enter a valid directory or file path. ⚠️")
        else:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            break

    while True:
        epochs = input("Please enter the number of epochs used for training (press enter to default to 50): ").strip()
        if epochs != "":
            try:
                epochs = int(epochs)
                assert isinstance(epochs, int)
                if epochs < 0:
                    print("⚠️ Invalid input. Epochs count must be greater than 0. ⚠️")
                else:
                    break
            except (ValueError, AssertionError):
                print("⚠️ Invalid integer. Please make enter a valid integer for the number of epochs. ⚠️")
        else:
            epochs = 50

    while True:
        batch_size = input("Please enter the batch size used for training (press enter to default to 16): ").strip()
        if batch_size != "":
            try:
                batch_size = int(batch_size)
                assert isinstance(batch_size, int)
                if batch_size < 0:
                    print("⚠️ Invalid input. Batch size must be greater than 0. ⚠️")
                else:
                    break
            except (ValueError, AssertionError):
                print("⚠️ Invalid integer. Please make enter a valid integer for the batch size. ⚠️")
        else:
            batch_size = 16
            break

    while True:
        cuda = input("Should we use CUDA for the training (y/n)? (press enter to default to no)").strip()
        if cuda != "":
            if cuda == "y":
                cuda = True
                break
            elif cuda == "n":
                cuda = False
                break
            else:
                print("⚠️ Invalid input. Please enter 'y' (yes) or 'n' (no) to enable Cuda "
                      "(press enter to default to no) ⚠️")
        else:
            cuda = False
            break

    TrainModel(train_path, val_path, checkpoint_path, epochs, batch_size, cuda)


def TrainModel(_train_folder, _val_folder, _checkpoint_path, _epochs, _batch_size, _cuda):

    # Create datasets and loaders.
    train_dataset = iDataLoader(_train_folder)
    val_dataset = iDataLoader(_val_folder)

    train_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=_batch_size, shuffle=False)

    # Device configuration
    device = torch.device("cuda" if (_cuda and torch.cuda.is_available()) else "cpu")
    model = iCNN().to(device)

    # Use Binary Cross Entropy Loss since all training examples are labeled as "good" (1.0).
    criterion = nn.BCELoss()

    # Use Adam optimizer.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch_loss = 0.0

    # Training loop
    for epoch in range(_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # Ensure shape is (batch, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch + 1}/{_epochs}], Loss: {epoch_loss:.4f}")

        # Validation loop (optional)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_dataset)
        print(f"Validation Loss: {val_loss:.4f}")

    checkpoint_file = os.path.join(_checkpoint_path, "reference_model.pth")

    # Save the trained model checkpoint.
    torch.save({
        'epoch': _epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, checkpoint_file)
    print(f"Model checkpoint saved at {checkpoint_file}")

    # cleanup
    train_dataset.DeleteItems()
    val_dataset.DeleteItems()