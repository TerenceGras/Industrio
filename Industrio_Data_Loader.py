import os
import shutil
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# -----------------------------------------
# Custom Transform for preprocessing images
# -----------------------------------------

# Define transformation once
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])


# -----------------------------------------
# Custom Dataset for loading pre-processed images
# -----------------------------------------
class iDataLoader(Dataset):
    def __init__(self, root_folder):

        self.image_files = []
        self.labels = []
        self.temp_ok = os.path.join(root_folder, "ok")
        self.temp_nok = os.path.join(root_folder, "nok")

        for label, subfolder in enumerate(["ok", "nok"]):  # "ok" -> 1.0, "nok" -> 0.0
            folder_path = os.path.join(root_folder, subfolder)
            if not os.path.exists(folder_path):
                continue  # Skip if folder doesn't exist

            new_dir = os.path.join(folder_path, "temp")

            # Remove the temp folder if it exists, then recreate it
            if os.path.exists(new_dir):
                shutil.rmtree(new_dir)  # Deletes all contents of the folder
            os.makedirs(new_dir)  # Create temp directory

            # Process images
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, img_name)
                    image = Image.open(img_path).convert('L')
                    image = transform(image)
                    save_path = os.path.join(new_dir, img_name + ".pt")
                    torch.save(image, save_path)

                    self.image_files.append(save_path)
                    self.labels.append(torch.tensor(float(label)))  # 1.0 for "ok", 0.0 for "nok"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = torch.load(self.image_files[idx])  # Load preprocessed image
        label = self.labels[idx]
        return image, label

    def DeleteItems(self):
        os.rmdir(os.path.join(self.temp_ok, "temp"))
        os.rmdir(os.path.join(self.temp_nok, "temp"))

