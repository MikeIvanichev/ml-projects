import matplotlib

matplotlib.use("Agg")  # Use the Agg backend for non-GUI environments
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as T
import pickle

with open("./data/dataset.pickle", "rb") as f:
    data = pickle.load(f)

train = data["train"]
val = data["val"]

# |%%--%%| <PZ4H4Rd197|1ExwV8HTSq>

print("Train set length: %i" % len(train))
print("Val set length: %i" % len(val))

# |%%--%%| <1ExwV8HTSq|bDxeXH0ddv>

import numpy as np


# Function to find the maximum RGB value across all images
def find_max_rgb_value(train):
    max_value = 0
    for item in train:
        image = item["image"]
        max_value = max(
            max_value, np.max(image)
        )  # Find the max value in the current image
    return max_value


# Find and print the maximum RGB value
max_rgb_value = find_max_rgb_value(train)
max_rgb_value = max(max_rgb_value, find_max_rgb_value(val))
print(f"The maximum RGB value in the dataset is: {max_rgb_value}")

# |%%--%%| <bDxeXH0ddv|gA4c4KSFIw>


# Function to plot pixel value distribution
def plot_pixel_value_distribution(train, val, max_rgb_value, save_path):
    # Collect pixel values from both train and validation datasets
    pixel_values = []

    for dataset in [train, val]:
        for item in dataset:
            image = item["image"]
            # Normalize the image to [0, 1] and collect pixel values
            norm_image = image / max_rgb_value
            pixel_values.append(norm_image.flatten())

    # Convert list to a single numpy array
    pixel_values = np.concatenate(pixel_values)

    # Plot the histogram of pixel values
    plt.figure(figsize=(10, 6))
    plt.hist(pixel_values, bins=100, color="blue", alpha=0.7)
    plt.title("Distribution of Pixel Values")
    plt.xlabel("Normalized Pixel Value")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Save the plot
    plt.savefig(save_path)
    plt.close()


# Specify the maximum RGB value and save path
max_rgb_value = 18640
save_path = "./pixel_value_distribution.png"

# Plot and save the distribution
plot_pixel_value_distribution(train, val, max_rgb_value, save_path)

# |%%--%%| <gA4c4KSFIw|rcafMgnwBa>

import numpy as np
import pandas as pd


# Function to calculate statistics for each image and save as a table
def save_image_statistics(train, val, save_path):
    # Calculate statistics for each image in both datasets
    def calculate_statistics(dataset, dataset_name):
        image_numbers = []
        avg_values = []
        min_values = []
        max_values = []
        spreads = []

        for idx, item in enumerate(dataset):
            image = item["image"]

            # Calculate statistics directly from pixel values
            avg_value = np.mean(image)
            min_value = np.min(image)
            max_value = np.max(image)
            spread = max_value - min_value

            # Append values to lists
            image_numbers.append(idx + 1)  # Image number (1-based index)
            avg_values.append(avg_value)
            min_values.append(min_value)
            max_values.append(max_value)
            spreads.append(spread)

        return image_numbers, avg_values, min_values, max_values, spreads

    (
        train_image_numbers,
        train_avg_values,
        train_min_values,
        train_max_values,
        train_spreads,
    ) = calculate_statistics(train, "Train")
    val_image_numbers, val_avg_values, val_min_values, val_max_values, val_spreads = (
        calculate_statistics(val, "Validation")
    )

    # Create DataFrames to store the results
    df_train = pd.DataFrame(
        {
            "Image Number": train_image_numbers,
            "Average Pixel Value": train_avg_values,
            "Min Pixel Value": train_min_values,
            "Max Pixel Value": train_max_values,
            "Spread": train_spreads,
            "Dataset": "Train",
        }
    )
    df_val = pd.DataFrame(
        {
            "Image Number": val_image_numbers,
            "Average Pixel Value": val_avg_values,
            "Min Pixel Value": val_min_values,
            "Max Pixel Value": val_max_values,
            "Spread": val_spreads,
            "Dataset": "Validation",
        }
    )

    # Concatenate DataFrames
    df = pd.concat([df_train, df_val], ignore_index=True)

    # Save DataFrame to a CSV file
    df.to_csv(save_path, index=False)


# Specify the save path
save_path = "./image_statistics.csv"

# Calculate and save the image statistics
save_image_statistics(train, val, save_path)


# |%%--%%| <rcafMgnwBa|oFT7ibWdov>

import numpy as np


# Function to calculate the ratio of 1s to 0s in the masks
def calculate_mask_ratio(dataset):
    total_ones = 0
    total_zeros = 0

    for item in dataset:
        mask = item["mask"]
        ones = np.sum(mask == 1)
        zeros = np.sum(mask == 0)
        total_ones += ones
        total_zeros += zeros

    # Compute the ratio of 1s to 0s
    if total_zeros == 0:  # Avoid division by zero
        ratio = float("inf")
    else:
        ratio = total_ones / total_zeros

    return ratio, total_ones, total_zeros


# Calculate the ratio for the train and val datasets
train_ratio, train_ones, train_zeros = calculate_mask_ratio(train)
val_ratio, val_ones, val_zeros = calculate_mask_ratio(val)

# Print the results
print(
    f"Train set - 1s to 0s ratio: {train_ratio:.4f} (Total 1s: {train_ones}, Total 0s: {train_zeros})"
)
print(
    f"Validation set - 1s to 0s ratio: {val_ratio:.4f} (Total 1s: {val_ones}, Total 0s: {val_zeros})"
)


# |%%--%%| <oFT7ibWdov|DdGhOClPJu>

import numpy as np


# Function to calculate the mean and std for each channel (B, G, R)
def calculate_mean_std(dataset, max_rgb_value):
    sum_channels = np.zeros(3)
    sum_sq_channels = np.zeros(3)
    num_pixels = 0

    for item in dataset:
        image = item["image"]
        # Normalize the image to the range [0, 1]
        norm_image = image / max_rgb_value

        # Sum the pixel values for each channel
        sum_channels += np.sum(norm_image, axis=(0, 1))

        # Sum of squares of pixel values for each channel
        sum_sq_channels += np.sum(norm_image**2, axis=(0, 1))

        # Update the total number of pixels
        num_pixels += norm_image.shape[0] * norm_image.shape[1]

    # Calculate mean for each channel
    mean = sum_channels / num_pixels

    # Calculate standard deviation for each channel
    std = np.sqrt(sum_sq_channels / num_pixels - mean**2)

    return mean, std


# Combine both train and val sets for global statistics
combined_dataset = train + val

# Specify the maximum RGB value
max_rgb_value = 8000

# Calculate mean and std for each channel
mean, std = calculate_mean_std(combined_dataset, max_rgb_value)

# Print results
print(f"Mean per channel (B, G, R): {mean}")
print(f"Standard deviation per channel (B, G, R): {std}")


# |%%--%%| <DdGhOClPJu|lUXezd6tPy>


def save_images_and_masks(train, save_dir, max_rgb_value):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the normalization transform
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for i in range(0, len(train), 3):
        fig, ax = plt.subplots(3, 2, figsize=(10, 15))

        for j in range(3):
            if i + j >= len(train):
                break
            item = train[i + j]
            image = item["image"]
            mask = item["mask"]

            # Normalize the image to the range [0, 1]
            norm_image = image / max_rgb_value
            norm_image[norm_image > 1] = 1

            # Convert from BGR to RGB
            norm_image_rgb = norm_image[..., ::-1]

            # Convert to tensor, apply normalization, and convert back to numpy array
            image_tensor = torch.tensor(norm_image_rgb.copy()).permute(2, 0, 1).float()
            # normalized_image = normalize(image_tensor).permute(1, 2, 0).numpy()
            normalized_image = image_tensor.permute(1, 2, 0).numpy()

            ax[j, 0].imshow(normalized_image)
            ax[j, 0].set_title(f"Image {i + j + 1}")
            ax[j, 0].axis("off")

            ax[j, 1].imshow(mask, cmap="gray")
            ax[j, 1].set_title(f"Mask {i + j + 1}")
            ax[j, 1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"image_mask_set_{i//3 + 1}.png"))
        plt.close()


# Specify the maximum RGB value in the dataset
max_rgb_value = 8000

# Specify the directory to save the plots
save_dir = "./output_images_train"
save_images_and_masks(train, save_dir, max_rgb_value)
save_dir = "./output_images_val"
save_images_and_masks(val, save_dir, max_rgb_value)
