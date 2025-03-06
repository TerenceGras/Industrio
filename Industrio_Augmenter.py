import os
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path


def PromptAugment():
    message = "💪 LET'S AUGMENT YOUR IMAGES FOR INDUSTRIO 💪"
    border = "*" * 50
    padding = (50 - len(message) - 2) // 2  # Calculate space padding

    print("\n" + border)
    print("*" + " " * padding + message + " " * (50 - len(message) - padding - 2) + "*")
    print(border + "\n")

    input_folder = ""
    while True:
        input_folder = input("Please enter the full path for the input folder: ").strip()
        if not os.path.exists(input_folder):  # Check if the path is valid
            print("⚠️ Invalid path. Please enter a valid directory or file path. ⚠️")
        else:
            break
    output_folder = ""
    while True:
        output_folder = input("Please enter the full path for the output folder: ").strip()
        try:
            Path(output_folder)  # Try creating a Path object
            break
        except Exception:
            print("⚠️ Invalid path. Please enter a valid directory or file path. ⚠️")

    """Process all images in input_folder, apply augmentation, and save them."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_images(input_folder, output_folder)


def noise_image(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.int16)  # Generate Gaussian noise
    noised_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)  # Add and clip
    return noised_image


def random_rotate_image(image):
    angle = random.choice([90, 180])  # Randomly choose 90 or 180 degrees
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:  # 180 degrees
        return cv2.rotate(image, cv2.ROTATE_180)


def random_flip(image):
    flip_type = random.choice([-1, 0, 1])  # -1 = both, 0 = vertical, 1 = horizontal
    return cv2.flip(image, flip_type)


def adjust_brightness(image, factor_range=(0.7, 1.3)):
    factor = random.uniform(*factor_range)
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def augment_image(image):
    return [image, random_rotate_image(image), random_flip(image), adjust_brightness(image),
            noise_image(image)]


def process_images(input_folder, output_folder):

    for img_name in os.listdir(input_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, img_name)

            image = cv2.imread(img_path)
            resized_image = cv2.resize(image, (416, 416))
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            augmented_images = augment_image(gray_image)

            for i, aug_img in enumerate(augmented_images):
                aug_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.png"
                aug_img_path = os.path.join(output_folder, aug_img_name)
                cv2.imwrite(aug_img_path, aug_img)

    print(f"✅ Augmentation complete! Augmented images are saved in '{output_folder}'. ✅")
