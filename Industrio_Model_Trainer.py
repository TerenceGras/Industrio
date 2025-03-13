import os
import torch
import sys
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Industrio_Data_Loader import iDataLoader
from Industrio_CNN import iCNN
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score


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
        model_name = input("Please enter the name of the model you are training: ").strip()
        if not model_name or model_name == "":  # Check if the path is valid
            print("⚠️ Invalid name. Please enter a valid file name. ⚠️")
        else:
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

    TrainModel(model_name, train_path, val_path, checkpoint_path, epochs, batch_size, cuda)


def warmup_lambda(epoch):
    if epoch < 5:
        return (epoch + 1) / 5  # Linear scaling
    return 1.0  # Keep LR constant after warmup


def TrainModel(_model_name, _train_folder, _val_folder, _checkpoint_path, _epochs, _batch_size, _cuda):
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
    optimizer = optim.AdamW(model.parameters(), lr=0.0019, weight_decay=1e-5)  # adding L2 weight decay
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0018, momentum=0.95, nesterov=True)
    epoch_loss = 0.0

    # Using a scheduler to reduce learning rate
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.9)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.0025, step_size_up=4000,
                                                  mode="triangular2")

    # Training loop
    final_train_loss = 0.0  # Store final training loss
    final_val_loss = 0.0  # Store final validation loss
    all_labels = []
    all_preds = []

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

        final_train_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch + 1}/{_epochs}], Loss: {final_train_loss:.4f}")

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

                # Store predictions and true labels for mAP calculation
                all_labels.extend(labels.cpu().numpy())  # Convert to numpy array
                all_preds.extend(outputs.cpu().numpy())  # Convert to numpy array

        final_val_loss = val_loss / len(val_dataset)
        scheduler.step()  # Reduce LR if val_loss stops improving
        print(f"Validation Loss: {final_val_loss:.4f}")

    # Convert collected predictions and labels to NumPy arrays
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()

    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    ap_score = average_precision_score(all_labels, all_preds)

    # Plot Precision-Recall curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, marker='.', label=f"AP = {ap_score:.4f}")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid()

    # Table data with training loss, validation loss, and mAP
    table_data = [
        ["Training Loss", "Validation Loss", "mAP"],
        [f"{final_train_loss:.4f}", f"{final_val_loss:.4f}", f"{ap_score:.4f}"]
    ]

    # Add table under the plot
    table = plt.table(cellText=table_data,
                      colLabels=None,
                      cellLoc='center',
                      loc='bottom',
                      bbox=[0.0, -0.3, 1.0, 0.2])  # Position table under the plot

    # Adjust layout so table doesn't overlap with the figure
    plt.subplots_adjust(bottom=0.35)

    # Save figure to checkpoint folder
    pr_curve_path = os.path.join(_checkpoint_path, f"{_model_name}_statistics.png")
    plt.savefig(pr_curve_path, bbox_inches='tight')
    plt.show()

    # Print final mAP score
    print(f"Final mean Average Precision (mAP): {ap_score:.4f}")
    print(f"Precision-Recall curve saved at: {pr_curve_path}")

    checkpoint_file = os.path.join(_checkpoint_path, f"{_model_name}.pth")

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
