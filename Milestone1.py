import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# ==============================================
#       HELPER FUNCTION: Display Images
# ==============================================
# (Keep the display_images function from previous responses)
def display_images(image_list, titles, figure_title="Image Processing Stages"):
    """Displays a list of images using matplotlib in a grid layout."""
    if not image_list: return
    num_images = len(image_list)
    if num_images == 0: return
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    plt.figure(figsize=(cols * 4, rows * 4.5))
    plt.suptitle(figure_title, fontsize=16)
    for i, (image, title) in enumerate(zip(image_list, titles)):
        plt.subplot(rows, cols, i + 1)
        if image is None:
            plt.title(f"{title}\n(Error)")
            plt.imshow(np.zeros((50, 50), dtype=np.uint8), cmap='gray', vmin=0, vmax=255)
        elif len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==============================================
#       STEP ONE FUNCTION (Using Sobel + Thresholding)
# ==============================================
def stepOne(image_path):
    """
    Performs Step 1: Preprocessing using Sobel edge detection followed by thresholding.

    Args:
        image_path (str): Path to the input image.

    Returns:
        tuple: (final_port_mask, intermediate_steps, titles)
    """
    print(f"--- Running Step 1 for: {os.path.basename(image_path)} ---")
    intermediate_steps = []
    titles = []

    # 1. Load Image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image {image_path}")
        return None, intermediate_steps, titles
    intermediate_steps.append(original_image.copy())
    titles.append("Original")

    # 2. Grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    intermediate_steps.append(gray_image.copy())
    titles.append("Grayscale")

    # 3. Blurring (Essential before Sobel to reduce noise sensitivity)
    # Use moderate blur
    blur_ksize = 7 # Must be odd
    blurred_image = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)
    # blurred_image = cv2.medianBlur(gray_image, blur_ksize) # Or try median
    intermediate_steps.append(blurred_image.copy())
    titles.append(f"Blurred (k={blur_ksize})")

    # --- 4. Sobel Edge Detection ---
    # Calculate gradients - use 64F for higher precision, then convert back
    grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=9) # ksize=3 or 5
    grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=9)

    # Calculate gradient magnitude
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # Scale magnitude to 0-255 range for visualization and thresholding
    # Using power scaling (e.g., sqrt) can sometimes enhance weaker edges visually
    # grad_magnitude_scaled = cv2.normalize(grad_magnitude**0.5, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # Linear scaling:
    grad_magnitude_scaled = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    intermediate_steps.append(grad_magnitude_scaled.copy())
    titles.append("Sobel Magnitude")
    # --- End Sobel ---

    # --- 5. Thresholding the SOBEL MAGNITUDE ---
    # Use Otsu to find a threshold for the edge strengths, or set a fixed one
    threshold_value, binary_edges = cv2.threshold(grad_magnitude_scaled, 0, 255,
                                                  cv2.THRESH_BINARY + int(abs(cv2.THRESH_OTSU * 1.1)))
    # Alternatively, try a fixed threshold (e.g., good edge strength value):
    # T_fixed = 50 # TUNE this value
    # _, binary_edges = cv2.threshold(grad_magnitude_scaled, T_fixed, 255, cv2.THRESH_BINARY)
    # threshold_value = T_fixed # For title

    intermediate_steps.append(binary_edges.copy())
    titles.append(f"Binary Edges (Otsu Thresh: {threshold_value:.0f})")
    # --- End Thresholding ---


    # --- 6. Morphological Cleanup (CRITICAL for edge maps) ---
    # We need to CONNECT the edges and FILL the shape. Closing is key.
    # Use larger kernels and more iterations than before.
    # Dilation first can help connect nearby edges before closing
    dilate_ksize = 3
    dilate_iter = 1
    kernel_dilate = np.ones((dilate_ksize, dilate_ksize), np.uint8)
    dilated_edges = cv2.dilate(binary_edges, kernel_dilate, iterations=dilate_iter)
    intermediate_steps.append(dilated_edges.copy())
    titles.append(f"Dilated Edges (k={dilate_ksize}, i={dilate_iter})")


    close_ksize = 7 # TUNE: Larger kernel needed to close gaps
    close_iter = 3 # TUNE: More iterations often needed
    kernel_close = np.ones((close_ksize, close_ksize), np.uint8)
    # Using closing on the DILATED edges
    closed_binary = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel_close, iterations=close_iter)
    intermediate_steps.append(closed_binary.copy())
    titles.append(f"Closed Edges (k={close_ksize}, i={close_iter})")
    # --- End Cleanup ---

    # 7. Contour Filtering (Applied to the filled shape)
    contours, hierarchy = cv2.findContours(closed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_port_mask = np.zeros_like(closed_binary)

    if not contours:
        print("Warning: No contours found after morphology.")
        intermediate_steps.append(final_port_mask)
        titles.append("Final Port Mask (Error)")
        return final_port_mask, intermediate_steps, titles

    # --- Contour Filtering Parameters (TUNING NEEDED) ---
    min_area = 200
    max_area = closed_binary.shape[0] * closed_binary.shape[1] * 0.8 # Allow larger area now
    min_aspect_ratio = 1.1
    max_aspect_ratio = 6.0
    # --- End Parameters ---

    potential_contours = []
    contour_viz = cv2.cvtColor(closed_binary, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            try:
                rect = cv2.minAreaRect(contour)
                (x, y), (width, height), angle = rect
                if width < 1 or height < 1: continue
                aspect_ratio = max(width / height, height / width)
                if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                     potential_contours.append(contour)
                     cv2.drawContours(contour_viz, [contour], -1, (0, 255, 0), 1) # Green
                else:
                     cv2.drawContours(contour_viz, [contour], -1, (255, 0, 0), 1) # Blue
            except Exception as e:
                cv2.drawContours(contour_viz, [contour], -1, (0, 0, 128), 1) # Brown
                continue
        else:
             cv2.drawContours(contour_viz, [contour], -1, (0, 0, 255), 1) # Red

    intermediate_steps.append(contour_viz)
    titles.append("Contour Filtering Viz")

    if not potential_contours:
        print("Warning: No contours passed filtering.")
        intermediate_steps.append(final_port_mask)
        titles.append("Final Port Mask (Error)")
        return final_port_mask, intermediate_steps, titles

    best_contour = max(potential_contours, key=cv2.contourArea)

    # 8. Create Final Mask
    cv2.drawContours(final_port_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    intermediate_steps.append(final_port_mask)
    titles.append("Final Port Mask")

    print("--- Step 1 Finished ---")
    return final_port_mask, intermediate_steps, titles

# ==============================================
#       MAIN EXECUTION (Example Usage)
# ==============================================
# (Keep the main execution block from previous responses, it should work with this stepOne)
if __name__ == "__main__":
    folder_path = input("Enter the path to the image folder (or press Enter to use test path): ")
    print(f"[DEBUG] Input received: '{folder_path}'")

    image_files_to_process = []

    if not folder_path:
        image_path_to_test = 'test_usb.jpg' # <---- ENSURE THIS FILE EXISTS or PROVIDE FULL PATH
        print(f"[DEBUG] -> No folder entered. Checking for default test image: {image_path_to_test}")
        if not os.path.isfile(image_path_to_test):
            print(f"!! ERROR: Default test image '{image_path_to_test}' not found or is not a file.")
            exit()
        print(f"[DEBUG] -> Default test file exists. Proceeding with this file.")
        image_files_to_process = [image_path_to_test]

    elif not os.path.isdir(folder_path):
        print(f"!! ERROR: Entered path '{folder_path}' is not a valid directory.")
        exit()

    else:
        print(f"[DEBUG] -> Entered path is a directory. Listing images in: {folder_path}")
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')
        try:
            all_files = os.listdir(folder_path)
            print(f"[DEBUG] -> Found {len(all_files)} total items in folder.")
            image_files_to_process = sorted([
                os.path.join(folder_path, f)
                for f in all_files
                if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(valid_extensions)
            ])
            print(f"[DEBUG] -> Found {len(image_files_to_process)} valid image files to process.")
        except Exception as e:
             print(f"!! ERROR listing files in directory '{folder_path}': {e}")
             exit()

        if not image_files_to_process:
             print(f"!! WARNING: No valid image files (with extensions {valid_extensions}) found in folder: {folder_path}")
             exit()

    print(f"\n>>> Starting processing for {len(image_files_to_process)} image(s)...")
    for img_path in image_files_to_process:
        print(f"\n--- Calling stepOne for: {img_path} ---")
        final_mask, steps, step_titles = stepOne(img_path)

        display_images(steps, step_titles, f"Step 1 Pipeline: {os.path.basename(img_path)}")

        if final_mask is not None:
            if cv2.countNonZero(final_mask) > 0:
                print(f"-> Successfully generated NON-EMPTY mask for {os.path.basename(img_path)}")
            else:
                print(f"-> Successfully generated EMPTY mask for {os.path.basename(img_path)} (Contour filtering likely failed)")
        else:
            print(f"-> Failed to generate mask for {os.path.basename(img_path)} (Error in stepOne)")
        print("-" * 50)

    print("\n>>> Finished processing all images.")
