import os
import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise

# Directory containing the dataset
dataset_dir = 'Sign_Language_Digits_Dataset'


# Function to add Gaussian noise to an image
def add_gaussian_noise(image_array, mean=0, var=0.01):
    noisy_image = random_noise(image_array, mode='gaussian', mean=mean, var=var)
    noisy_image = (255 * noisy_image).astype(np.uint8)
    return noisy_image


# Function to scale an image
def scale_image(image, scale_factor):
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    scaled_image = image.resize((new_width, new_height), Image.LANCZOS)
    return scaled_image


# Function to augment images in a directory
def augment_images(input_dir, output_dir, noise_var=0.01, scale_factors=[0.9, 1.1]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        output_label_dir = os.path.join(output_dir, label)
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)

        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            image = Image.open(image_path)
            image_array = np.array(image)

            # Add Gaussian noise
            noisy_image_array = add_gaussian_noise(image_array, var=noise_var)
            noisy_image = Image.fromarray(noisy_image_array)
            noisy_image.save(os.path.join(output_label_dir, f'noisy_{image_name}'))

            # Scale images
            for scale_factor in scale_factors:
                scaled_image = scale_image(image, scale_factor)
                # Save scaled image
                scaled_image_name = f'scaled_{scale_factor}_{image_name}'
                scaled_image.save(os.path.join(output_label_dir, scaled_image_name))


# Apply data augmentation
augment_images(dataset_dir, 'Augmented_Sign_Language_Digits_Dataset')
