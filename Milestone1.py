import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_image(image_path):
    """Applies preprocessing steps to a USB connector image."""

    original_image = cv2.imread(image_path)
    processed_images = [original_image.copy()]
    titles = ["Original Image"]

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_images.append(gray_image.copy())
    titles.append("Grayscale")

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    processed_images.append(blurred_image.copy())
    titles.append("Blurred")

    # 4. Image Enhancement (Sharpening)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(blurred_image, -1, kernel)
    processed_images.append(sharpened_image.copy())
    titles.append("Sharpened")

    # 5. Binarization (Otsu's Thresholding)
    _, binary_image = cv2.threshold(sharpened_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(binary_image.copy())
    titles.append("Binary")

    # 6. Geometric Transformations (Placeholder)
    rotated_image = binary_image.copy()  # Replace with actual rotation correction
    processed_images.append(rotated_image.copy())
    titles.append("Rotated")

    return processed_images, titles


def extract_features(binary_image):
    """Extracts features from the preprocessed binary image."""

    feature_images = []
    feature_titles = []

    # 1. Edge Detection (Canny)
    edges_canny = cv2.Canny(binary_image, 100, 200)  # Tune thresholds as needed
    feature_images.append(edges_canny.copy())
    feature_titles.append("Canny Edges")

    return feature_images, feature_titles


def postprocess_canny_edges(canny_image, original_binary_image):
    """
    Further processes the Canny edge image to isolate the USB connector.

    Args:
        canny_image (numpy.ndarray): The Canny edge image.
        original_binary_image (numpy.ndarray): The original binary image
                                             (for masking).

    Returns:
        numpy.ndarray: The processed Canny edge image.
    """

    # 1. Dilate the Canny edges to connect broken edges
    dilated_edges = cv2.dilate(canny_image, None, iterations=2)
    processed_images = [dilated_edges.copy()]
    titles = ["Dilated Edges"]

    # 2. Find contours in the dilated edge image
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    processed_images.append(dilated_edges.copy())
    titles.append("Contour Lines")

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Assume the largest contour is the USB connector
    usb_contour = contours[0]

    # 3. Create a mask from the USB contour
    mask = np.zeros_like(canny_image)
    cv2.drawContours(mask, [usb_contour], -1, 255, thickness=cv2.FILLED)
    processed_images.append(mask.copy())
    titles.append("Mask")

    # 4. Apply the mask to the original Canny edge image
    isolated_canny_edges = cv2.bitwise_and(canny_image, canny_image, mask=mask)
    processed_images.append(isolated_canny_edges.copy())
    titles.append("Isolated Canny Edges")

    return processed_images, titles


def display_images(image_list, titles):
    """Displays a list of images using matplotlib."""

    plt.figure(figsize=(15, 5 * len(image_list)))
    for i, (image, title) in enumerate(zip(image_list, titles)):
        plt.subplot(1, len(image_list), i + 1)
        if len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


def save_images(image_list, titles, output_dir):
    """Saves a list of images to the specified directory."""

    os.makedirs(output_dir, exist_ok=True)
    for i, (image, title) in enumerate(zip(image_list, titles)):
        filename = f"{title.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        if len(image.shape) == 3:
            cv2.imwrite(filepath, image)
        else:
            cv2.imwrite(filepath, image)
    print(f"Saved images to: {output_dir}")


# --- Main Script ---
image_path = input("Enter the path to the USB image: ")
output_dir = "output_images"  # Directory to save output images

# Step 1: Image Preprocessing
preprocessed_images, preprocessed_titles = preprocess_image(image_path)
display_images(preprocessed_images, preprocessed_titles)
save_images(preprocessed_images, preprocessed_titles, output_dir)

# Step 2: Feature Extraction
binary_image = preprocessed_images[5]  # The binary image from preprocessing
canny_images, canny_titles = extract_features(binary_image)
display_images(canny_images, canny_titles)
save_images(canny_images, canny_titles, output_dir)
canny_image = canny_images[0]

# Post-process Canny edges to isolate USB
postprocessed_images, postprocessed_titles = postprocess_canny_edges(canny_image, binary_image)
display_images(postprocessed_images, postprocessed_titles)
save_images(postprocessed_images, postprocessed_titles, output_dir)