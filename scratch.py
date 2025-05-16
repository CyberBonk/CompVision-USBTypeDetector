import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# ==============================================
#       TUNING CONSTANTS FOR DYNAMIC MORPHOLOGY
# ==============================================
# Thickness thresholds (adjust based on typical edge thickness in pixels)
THIN_EDGE_MAX = 4
MEDIUM_EDGE_MAX = 6

# Parameters for THIN edges
K_OPEN_THIN = 3 # Small kernel to remove noise but keep thin lines (must be odd)
IT_OPEN_THIN = 1
K_CLOSE_THIN = 5 # Moderate kernel to close small gaps (must be odd)
IT_CLOSE_THIN = 1

# Parameters for MEDIUM edges
K_OPEN_MEDIUM = 5 # Can still be relatively small (must be odd)
IT_OPEN_MEDIUM = 1
K_CLOSE_MEDIUM = 9 # Larger kernel to close bigger gaps (must be odd)
IT_CLOSE_MEDIUM = 2

# Parameters for THICK edges
K_OPEN_THICK = 5  # Larger kernel ok if edges are thick (must be odd)
IT_OPEN_THICK = 1
K_CLOSE_THICK = 13 # Significantly larger kernel needed (must be odd)
IT_CLOSE_THICK = 2
# ==============================================

# ==============================================
#       HELPER FUNCTION: Display Images
# ==============================================
def display_images(image_list, titles, figure_title="Image Processing Stages"):
    """Displays a list of images using matplotlib in a grid layout."""
    if not image_list: return
    num_images = len(image_list)
    if num_images == 0: return
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    if rows < cols and num_images > 4: cols = math.ceil(num_images / rows)
    fig_width = max(12, cols * 3)
    fig_height = max(4, rows * 3.5)
    plt.figure(figsize=(fig_width, fig_height))
    plt.suptitle(figure_title, fontsize=14)
    for i, (image, title) in enumerate(zip(image_list, titles)):
        plt.subplot(rows, cols, i + 1)
        if image is None:
            plt.title(f"{title}\n(Error)")
            plt.imshow(np.zeros((50, 50), dtype=np.uint8), cmap='gray', vmin=0, vmax=255)
        elif len(image.shape) == 3:
            display_img = image.astype(np.uint8)
            plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        else:
            display_img = image.astype(np.uint8)
            plt.imshow(display_img, cmap='gray')
        plt.title(title, fontsize=9)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.show()


# ==============================================
#       HELPER FUNCTION: Estimate Edge Thickness
# ==============================================
def estimate_edge_thickness(binary_edges):
    """
    Estimates the median thickness of white edges on a black background.

    Args:
        binary_edges: Input binary image (edges are white).

    Returns:
        int: Estimated median edge thickness (at least 1).
    """
    if binary_edges is None or cv2.countNonZero(binary_edges) == 0:
        return 1 # Default thickness if no edges

    try:
        # Invert: Distance transform finds distance to nearest zero (black) pixel
        inv = cv2.bitwise_not(binary_edges)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3) # Euclidean distance, 3x3 mask

        # Thickness is roughly 2 * distance for pixels within the original edges
        # Consider only distances > 0
        dist_on_edges = dist[binary_edges > 0] # Get distances only where original edges were white
        if dist_on_edges.size == 0:
             # This can happen if edges are perfectly 1 pixel thin near border
             # or if distance transform failed unexpectedly
             # Alternative: check distances > 0 in the whole dist map?
             dist_on_edges = dist[dist > 0]
             if dist_on_edges.size == 0: return 1


        thicknesses = (dist_on_edges * 2).astype(int)

        if thicknesses.size == 0:
            return 1 # Should not happen if dist_on_edges had elements, but safeguard

        # Use median for robustness against outliers
        median_thickness = int(np.median(thicknesses))

        return max(median_thickness, 1) # Ensure thickness is at least 1
    except Exception as e:
        print(f"Error in estimate_edge_thickness: {e}")
        return 1 # Default on error


# ==============================================
#       DESKEW FUNCTION (Canny on Gray + Hough)
# ==============================================
# (Keep get_rotation_angle_and_deskew function - unchanged from previous)
def get_rotation_angle_and_deskew(gray_image, angle_tolerance=25,
                                  canny_thresh1=50, canny_thresh2=150,
                                  hough_thresh=40, min_line_len=40, max_line_gap=10):
    """Calculates rotation angle using Canny(Gray)+Hough and rotates the gray image."""
    print("--- Calculating Rotation Angle (Canny(Gray) + Hough) ---")
    steps = []
    titles = []
    h, w = gray_image.shape[:2]
    rotated_image = gray_image.copy()
    calculated_angle = 0.0
    blur_ksize = 3
    blurred_for_canny = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)
    steps.append(blurred_for_canny.copy()); titles.append(f"Blur for Canny (k={blur_ksize})")
    edges = cv2.Canny(blurred_for_canny, canny_thresh1, canny_thresh2)
    steps.append(edges.copy()); titles.append(f"Canny Edges (T:{canny_thresh1}-{canny_thresh2})")
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=hough_thresh,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    lines_img_viz = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR); angles = []
    if lines is not None:
        #print(f"[DEBUG HOUGH] Found {len(lines)} lines.")
        for line in lines:
            x1, y1, x2, y2 = line[0]; cv2.line(lines_img_viz, (x1, y1), (x2, y2), (0, 0, 255), 1)
            dx, dy = x2 - x1, y2 - y1;
            if dx == 0: continue
            ang = np.degrees(np.arctan2(dy, dx))
            if abs(ang) > (180 - angle_tolerance): ang = ang - math.copysign(180.0, ang)
            if abs(ang) <= angle_tolerance:
                angles.append(ang); cv2.line(lines_img_viz, (x1, y1), (x2, y2), (0, 255, 0), 1)
    else: print("Warning: HoughLinesP found no lines.")
    steps.append(lines_img_viz); titles.append("Hough Lines Viz")
    if angles:
        calculated_angle = np.mean(angles); print(f"Average Horizontal Angle: {calculated_angle:.2f} degrees")
        min_angle_thresh = 0.5
        if abs(calculated_angle) > min_angle_thresh:
            center = (w // 2, h // 2); M = cv2.getRotationMatrix2D(center, calculated_angle, 1.0)
            rotated_image = cv2.warpAffine(gray_image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            print(f"Applied rotation of {calculated_angle:.2f} degrees.")
        else: print(f"Rotation skipped, angle {calculated_angle:.2f} below threshold."); rotated_image = gray_image.copy()
    else: print("Warning: No lines for angle calculation. Rotation skipped."); rotated_image = gray_image.copy()
    print("--- Deskew Calculation Finished ---")
    return rotated_image, calculated_angle, steps, titles


# ==============================================
#       STEP ONE FUNCTION (Integrates Deskew + Dynamic Morph)
# ==============================================
def stepOne(image_path):
    """
    Performs Step 1: Deskews, applies Sobel, thresholds Sobel Mag with Otsu,
    cleans with DYNAMIC morph ops and connected components, generates final mask.
    """
    print(f"--- Running Full Step 1 for: {os.path.basename(image_path)} ---")
    intermediate_steps = []
    titles = []
    final_port_mask = None

    # --- Initial Load and Grayscale ---
    original_image = cv2.imread(image_path)
    if original_image is None: return None, intermediate_steps, titles
    intermediate_steps.append(original_image.copy()); titles.append("Original")
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    intermediate_steps.append(gray_image.copy()); titles.append("Grayscale")
    # --- End Initial Load ---

    # --- Deskewing Step ---
    try:
         rotated_gray, angle, deskew_steps_list, deskew_titles_list = get_rotation_angle_and_deskew(
             gray_image.copy(),
             # --- TUNE DESKEW PARAMS ---
             angle_tolerance=25, hough_thresh=40, min_line_len=30, max_line_gap=10
         )
         intermediate_steps.extend(deskew_steps_list); titles.extend(deskew_titles_list)
         intermediate_steps.append(rotated_gray.copy()); titles.append(f"Rotated Gray (Angle: {angle:.2f})")
    except Exception as e:
         print(f"!! ERROR during deskewing: {e}")
         intermediate_steps.append(gray_image.copy()); titles.append("Deskew Failed")
         rotated_gray = gray_image # Use original gray if deskew fails
    # --- End Deskewing Step ---

    # --- Process the ROTATED Grayscale Image ---
    print("--- Processing Rotated Image for Mask ---")
    input_for_processing = rotated_gray

    blur_ksize = 5 # TUNE
    blurred_rotated = cv2.GaussianBlur(input_for_processing, (blur_ksize, blur_ksize), 0)
    intermediate_steps.append(blurred_rotated.copy()); titles.append(f"Blurred Rotated (k={blur_ksize})")

    sobel_ksize = 9
    grad_x = cv2.Sobel(blurred_rotated, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    grad_y = cv2.Sobel(blurred_rotated, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)
    grad_magnitude_scaled = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    intermediate_steps.append(grad_magnitude_scaled.copy()); titles.append(f"Sobel Mag (Rotated, k={sobel_ksize})")

    threshold_value, binary_edges = cv2.threshold(grad_magnitude_scaled, 0, 255,
                                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    intermediate_steps.append(binary_edges.copy()); titles.append(f"Binary Edges (Otsu Th: {threshold_value:.0f})")


    # --- DYNAMIC Morphological Cleanup on Binary Edges ---
    print("--- Applying Dynamic Morphological Cleanup ---")
    median_thick = estimate_edge_thickness(binary_edges)
    print(f"Estimated Median Edge Thickness: {median_thick}")

    # Select parameters based on thickness
    if median_thick <= THIN_EDGE_MAX:
        k_open, it_open = K_OPEN_THIN, IT_OPEN_THIN
        k_close, it_close = K_CLOSE_THIN, IT_CLOSE_THIN
        print(f"Using THIN parameters (k_open={k_open}, it={it_open} / k_close={k_close}, it={it_close})")
    elif median_thick <= MEDIUM_EDGE_MAX:
        k_open, it_open = K_OPEN_MEDIUM, IT_OPEN_MEDIUM
        k_close, it_close = K_CLOSE_MEDIUM, IT_CLOSE_MEDIUM
        print(f"Using MEDIUM parameters (k_open={k_open}, it={it_open} / k_close={k_close}, it={it_close})")
    else: # Thick
        k_open, it_open = K_OPEN_THICK, IT_OPEN_THICK
        k_close, it_close = K_CLOSE_THICK, IT_CLOSE_THICK
        print(f"Using THICK parameters (k_open={k_open}, it={it_open} / k_close={k_close}, it={it_close})")

    # Apply Opening
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
    opened_binary = cv2.morphologyEx(binary_edges, cv2.MORPH_OPEN, kernel_open, iterations=it_open)
    intermediate_steps.append(opened_binary.copy())
    titles.append(f"Opened Edges (k={k_open}, i={it_open})")

    # Apply Closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
    closed_binary = cv2.morphologyEx(opened_binary, cv2.MORPH_CLOSE, kernel_close, iterations=it_close)
    intermediate_steps.append(closed_binary.copy())
    titles.append(f"Closed Edges (k={k_close}, i={it_close})")
    # --- End Dynamic Morph Cleanup ---

    # --- Connected Components Analysis ---
    final_port_mask = np.zeros_like(closed_binary)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_binary, 4, cv2.CV_32S)

    if num_labels <= 1:
        print("Warning: No components found after morphology.")
        intermediate_steps.append(closed_binary); titles.append("Components (None Found)")
        intermediate_steps.append(final_port_mask); titles.append("Final Port Mask (Error)")
        return final_port_mask, intermediate_steps, titles
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        print(f"Largest component label: {largest_label}, Area: {stats[largest_label, cv2.CC_STAT_AREA]}")

        largest_comp_img = np.zeros_like(closed_binary)
        largest_comp_img[labels == largest_label] = 255
        intermediate_steps.append(largest_comp_img.copy())
        titles.append(f"Largest Component (Label {largest_label})")

        # --- Convex Hull ---
        contours, _ = cv2.findContours(largest_comp_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best_contour = contours[0]
            hull = cv2.convexHull(best_contour)
            cv2.drawContours(final_port_mask, [hull], -1, 255, thickness=cv2.FILLED)
            intermediate_steps.append(final_port_mask.copy())
            titles.append("Final Port Mask (Hull)")
        else:
            print("Warning: Could not find contour for largest component.")
            intermediate_steps.append(final_port_mask.copy())
            titles.append("Final Port Mask (Contour Error)")

    print("--- Step 1 Finished ---")
    return final_port_mask, intermediate_steps, titles

# ==============================================
#       MAIN EXECUTION (Using "Images" Subfolder)
# ==============================================
# (Keep the main execution block - unchanged from previous)
if __name__ == "__main__":
    try: script_dir = os.path.dirname(__file__)
    except NameError: script_dir = os.getcwd()
    folder_path = os.path.join(script_dir, "Images")
    print(f"--- Attempting to process images in: '{folder_path}' ---")
    image_files_to_process = []
    if not os.path.isdir(folder_path): print(f"!! ERROR: Subfolder '{folder_path}' not found."); exit()
    else:
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        try:
            all_files = os.listdir(folder_path)
            image_files_to_process = sorted([ os.path.join(folder_path, f)
                for f in all_files if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(valid_extensions)])
            print(f"[INFO] Found {len(image_files_to_process)} valid image files to process.")
        except Exception as e: print(f"!! ERROR listing files: {e}"); exit()
        if not image_files_to_process: print(f"!! WARNING: No valid image files found."); exit()

    print(f"\n>>> Starting processing...")
    for img_path in image_files_to_process:
        print(f"\n--- Calling stepOne for: {os.path.basename(img_path)} ---")
        final_mask, steps, step_titles = stepOne(img_path)
        display_images(steps, step_titles, f"Step 1 Full Pipeline: {os.path.basename(img_path)}")
        if final_mask is not None:
            status = "NON-EMPTY" if cv2.countNonZero(final_mask) > 0 else "EMPTY"
            print(f"-> Successfully generated {status} mask for {os.path.basename(img_path)}")
            print("   (Next steps would process 'Rotated Gray' and/or 'Final Port Mask')")
        else: print(f"-> Failed Step 1 processing for {os.path.basename(img_path)}")
        print("-" * 50)
    print("\n>>> Finished processing all images.")