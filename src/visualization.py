import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def denormalize_image(tensor: torch.Tensor):
    """
    Converts a normalized PyTorch tensor image into a denormalized NumPy array 
    suitable for visualization.
    The input tensor is expected to have the shape [C, H, W] (channel-first format),
    and it should be normalized using the ImageNet mean and standard deviation values.
    Args:
        tensor (torch.Tensor): A PyTorch tensor representing the image, 
                               normalized with ImageNet mean and std.
    Returns:
        numpy.ndarray: A denormalized image in HWC format (height, width, channels) 
                       with pixel values scaled to the range [0, 255] and dtype uint8.
    """
    # Tensor images in PyTorch are [C x H x W]
    # Imagenet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Denormalization: multiply by std and sum mean
    image_np = tensor.cpu().detach().squeeze().permute(1, 2, 0).numpy()  #Convert tu numpy HWC
    image_np = image_np * std + mean  
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)  # Scale to rage 0-255 and convert to uint8
    
    return image_np


def visualize_data(data, n_imgs, rows, is_batch=False):
    """
    Visualize images and masks from a dataset or DataLoader batch.
    
    Parameters:
    - data: Dataset object or DataLoader batch (tuple of images, masks, names).
    - n_imgs: Number of images and masks to visualize.
    - rows: Number of rows in the visualization grid.
    - is_batch: Boolean flag indicating if data comes from a batch (default: False).
    """
    # Handle batch or dataset input
    if is_batch:
        batch_images, batch_masks, batch_names = next(iter(data))
        images = batch_images[:n_imgs]
        masks = batch_masks[:n_imgs]
        names = batch_names[:n_imgs]
    else:
        # Random indices for plotting
        indices = [random.randint(0, len(data) - 1) for _ in range(n_imgs)]
        images, masks, names = zip(*[data[idx] for idx in indices])
   
    # Create two figures for images and masks
    fig_images = plt.figure(figsize=(20, 10))
    fig_masks = plt.figure(figsize=(20, 10))
    
    for i in range(n_imgs):
        # Retrieve image and mask
        img_tensor = images[i]
        mask_tensor = masks[i]
        img_name = names[i]
          
        # Convert image and mask to numpy arrays for visualization
        img = denormalize_image(img_tensor) if is_batch else img_tensor.permute(1, 2, 0).numpy()  # CHW to HWC
        mask = mask_tensor.numpy()  # Masks are grayscale
    
        # Add image to the first figure
        plt.figure(fig_images.number)
        plt.subplot(rows, n_imgs // rows, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Image {img_name}")
        
        # Add mask to the second figure
        plt.figure(fig_masks.number)
        plt.subplot(rows, n_imgs // rows, i + 1)
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        plt.title(f"Mask {img_name}")
    
    # Show both figures
    fig_images.suptitle("Images", fontsize=16)
    fig_masks.suptitle("Masks", fontsize=16)
    fig_images.tight_layout()
    fig_masks.tight_layout()
    plt.show()


def visualize_results(image_path, mask_path, prediction):
    """
    Visualize the original image, true segmentation mask, and predicted segmentation mask side by side.
    Args:
        image_path (str): Path to the original image file.
        mask_path (str): Path to the true segmentation mask file.
        prediction (numpy.ndarray): Predicted segmentation mask as a 2D array.
    Returns:
        None: This function displays the visualizations using matplotlib and does not return any value.
    """
    original_image = Image.open(image_path)
    true_mask = Image.open(mask_path)
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")
    
    # True mask
    plt.subplot(1, 3, 2)
    plt.title("True Segmentation")
    plt.imshow(true_mask, cmap="gray")
    plt.axis("off")
    
    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.title("Predicted Segmentation")
    plt.imshow(prediction, cmap="gray")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()