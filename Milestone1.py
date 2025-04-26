import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_image(image_path):
    """
    Applies preprocessing steps to a USB connector image.

    Args:
        image_path (str): The path to the input image.

    Returns:
        list: A list of processed images at each step.
    """

    # 1. Load the image
    original_image = cv2.imread(image_path)
    processed_images = [original_image.copy()]  # Store original image

    # 2. Convert to Grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_images.append(gray_image.copy())

    # 3. Noise Reduction
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Example: 5x5 Gaussian blur
    processed_images.append(blurred_image.copy())

    # 4. Image Enhancement (Sharpening)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(blurred_image, -1, kernel)
    processed_images.append(sharpened_image.copy())

    # 5. Binarization (Otsu's Thresholding)
    _, binary_image = cv2.threshold(sharpened_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(binary_image.copy())

    # 6. Geometric Transformations (Basic Example: Rotation Correction)
    #   This is a placeholder.  Accurate rotation correction needs more advanced logic
    #   (e.g., Hough lines to detect the USB's main axis).
    rotated_image = binary_image  # Replace with actual rotation correction
    processed_images.append(rotated_image.copy())

    return processed_images


def display_images(image_list, titles):
    """
    Displays a list of images using matplotlib.

    Args:
        image_list (list): A list of images to display.
        titles (list): A list of titles for each image.
    """

    plt.figure(figsize=(15, 5 * len(image_list)))  # Adjust figure size as needed
    for i, (image, title) in enumerate(zip(image_list, titles)):
        plt.subplot(1, len(image_list), i + 1)
        if len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Color image
        else:
            plt.imshow(image, cmap='gray')  # Grayscale image
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

def save_images(image_list, titles, output_dir):
    """
    Saves a list of images to the specified directory.

    Args:
        image_list (list): A list of images to save.
        titles (list): A list of titles for each image (used in filenames).
        output_dir (str): The directory to save the images to.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    for i, (image, title) in enumerate(zip(image_list, titles)):
        filename = f"{title.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        if len(image.shape) == 3:
            cv2.imwrite(filepath, image)  # Color image
        else:
            cv2.imwrite(filepath, image)  # Grayscale image
    print(f"Saved images to: {output_dir}")

# --- Main Script ---
image_path = input("Enter the path to the USB image: ")

processed_images = preprocess_image(image_path)

titles = ["Original Image", "Grayscale", "Blurred", "Sharpened", "Binary", "Rotated"]

display_images(processed_images, titles)

output_dir = "output_images"
save_images(processed_images, titles, output_dir)