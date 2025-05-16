import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# ==============================================
# TUNING CONSTANTS FOR DYNAMIC MORPHOLOGY
THIN_EDGE_MAX    = 4
MEDIUM_EDGE_MAX  = 6
# thin edges
K_OPEN_THIN      = 3
IT_OPEN_THIN     = 1
K_CLOSE_THIN     = 5
IT_CLOSE_THIN    = 1
# medium edges
K_OPEN_MEDIUM    = 5
IT_OPEN_MEDIUM   = 1
K_CLOSE_MEDIUM   = 9
IT_CLOSE_MEDIUM  = 2
# thick edges
K_OPEN_THICK     = 5
IT_OPEN_THICK    = 1
K_CLOSE_THICK    = 13
IT_CLOSE_THICK   = 2


# ==============================================
# DISPLAY A GRID OF IMAGES (for debugging)
def display_images(image_list, titles, figure_title="Stages"):
    if not image_list:
        return
    n    = len(image_list)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n/cols)
    plt.figure(figsize=(cols*3, rows*3))
    plt.suptitle(figure_title, fontsize=14)
    for i, (img, ttl) in enumerate(zip(image_list, titles)):
        plt.subplot(rows, cols, i+1)
        if img is None:
            plt.imshow(np.zeros((50,50),np.uint8), cmap='gray')
        elif img.ndim==3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(ttl, fontsize=9)
        plt.xticks([]); plt.yticks([])
    plt.tight_layout(rect=[0,0.02,1,0.94])
    plt.show()


# ==============================================
# ROBUST MEDIAN EDGE-THICKNESS ESTIMATE
def estimate_edge_thickness(binary_edges):
    if binary_edges is None or cv2.countNonZero(binary_edges)==0:
        return 1
    try:
        inv  = cv2.bitwise_not(binary_edges)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        vals = dist[binary_edges>0]
        if vals.size==0:
            vals = dist[dist>0]
        if vals.size==0:
            return 1
        thickness = (vals*2).astype(int)
        lo, hi = np.percentile(thickness, [5,95])
        inliers = thickness[(thickness>=lo)&(thickness<=hi)]
        med = int(np.median(inliers)) if inliers.size else int(np.median(thickness))
        return max(1, med)
    except:
        return 1


# ==============================================
# DESKEW VIA CANNY + HOUGH LINES
def get_rotation_angle_and_deskew(
    gray_image,
    angle_tolerance=25,
    canny_thresh1=50, canny_thresh2=150,
    hough_thresh=40,
    min_line_len=40, max_line_gap=10
):
    steps, titles = [], []
    h, w = gray_image.shape[:2]

    # 1) small blur → Canny edges
    blur3 = cv2.GaussianBlur(gray_image, (3,3), 0)
    steps.append(blur3.copy()); titles.append("Blur for Canny (3×3)")
    edges = cv2.Canny(blur3, canny_thresh1, canny_thresh2)
    steps.append(edges.copy()); titles.append(f"Canny Edges ({canny_thresh1}-{canny_thresh2})")

    # 2) HoughLinesP visualization
    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi/180,
        threshold=hough_thresh,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )
    viz = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    angles = []
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            cv2.line(viz, (x1,y1), (x2,y2), (0,0,255), 1)
            dx, dy = x2-x1, y2-y1
            if dx==0: continue
            ang = math.degrees(math.atan2(dy,dx))
            if abs(ang)>(180-angle_tolerance):
                ang -= math.copysign(180, ang)
            if abs(ang)<=angle_tolerance:
                angles.append(ang)
                cv2.line(viz, (x1,y1), (x2,y2), (0,255,0), 1)
    steps.append(viz.copy()); titles.append("Hough Lines Viz")

    # 3) rotate if enough consistent lines
    rotated = gray_image.copy()
    if angles:
        avg_ang = float(np.mean(angles))
        if abs(avg_ang)>0.5:
            M = cv2.getRotationMatrix2D((w//2, h//2), avg_ang, 1.0)
            rotated = cv2.warpAffine(
                gray_image, M, (w,h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

    return rotated, angles, steps, titles


# ==============================================
# STEP ONE: FULL PIPELINE (largest CC as final mask)
def stepOne(image_path):
    print(f"\n--- Processing: {os.path.basename(image_path)} ---")
    intermediate_steps = []
    titles = []

    # 1) load & original
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, intermediate_steps, titles
    intermediate_steps.append(img_bgr.copy()); titles.append("Original")

    # 2) grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    intermediate_steps.append(gray.copy()); titles.append("Grayscale")

    # 3) deskew
    gray_rot, angles, ds, dtitles = get_rotation_angle_and_deskew(
        gray,
        angle_tolerance=25,
        canny_thresh1=50, canny_thresh2=150,
        hough_thresh=40,
        min_line_len=30, max_line_gap=10
    )
    intermediate_steps.extend(ds); titles.extend(dtitles)
    intermediate_steps.append(gray_rot.copy()); titles.append("Rotated Gray")

    # 4) blur rotated
    blur5 = cv2.GaussianBlur(gray_rot, (5,5), 0)
    intermediate_steps.append(blur5.copy()); titles.append("Blurred Rotated (5×5)")

    # 5) Canny edges
    canny_low, canny_high = 40, 100
    edges_canny = cv2.Canny(blur5, canny_low, canny_high)
    intermediate_steps.append(edges_canny.copy())
    titles.append(f"Canny Edges ({canny_low}-{canny_high})")

    # 6) dilate
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thick_edges = cv2.dilate(edges_canny, dilate_kernel, iterations=1)
    intermediate_steps.append(thick_edges.copy()); titles.append("Dilated Edges (3×3)")

    # 7) morphological cleanup
    median_thick = estimate_edge_thickness(thick_edges)
    print(f" → median edge thickness ≃ {median_thick}")
    if median_thick<=THIN_EDGE_MAX:
        ko,io = K_OPEN_THIN, IT_OPEN_THIN
        kc,ic = K_CLOSE_THIN, IT_CLOSE_THIN
    elif median_thick<=MEDIUM_EDGE_MAX:
        ko,io = K_OPEN_MEDIUM, IT_OPEN_MEDIUM
        kc,ic = K_CLOSE_MEDIUM, IT_CLOSE_MEDIUM
    else:
        ko,io = K_OPEN_THICK, IT_OPEN_THICK
        kc,ic = K_CLOSE_THICK, IT_CLOSE_THICK

    opened = cv2.morphologyEx(
        thick_edges,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (ko,ko)),
        iterations=io
    )
    intermediate_steps.append(opened.copy()); titles.append(f"Opened (k={ko},i={io})")

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kc,kc)),
        iterations=ic
    )
    intermediate_steps.append(closed.copy()); titles.append(f"Closed (k={kc},i={ic})")

    # 8) extract largest connected component
    final_mask = np.zeros_like(closed)
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8, ltype=cv2.CV_32S
    )
    if n_lbl > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        big   = np.argmax(areas) + 1
        comp  = (labels == big).astype(np.uint8) * 255
        final_mask = comp
        intermediate_steps.append(comp.copy()); titles.append(f"Largest CC ({big})")
    else:
        # if only background or 1 label, leave empty mask
        intermediate_steps.append(final_mask.copy()); titles.append("Largest CC (none)")

    return final_mask, intermediate_steps, titles


# ==============================================
# MAIN: Process & Save the largest CC into output_image
if __name__ == "__main__":
    base_dir    = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    images_dir  = os.path.join(base_dir, "Images")
    output_dir  = os.path.join(base_dir, "output_image")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(images_dir):
        print(f"ERROR: '{images_dir}' not found.")
        exit(1)

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = sorted([
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith(valid_exts)
    ])
    print(f"Found {len(files)} images in '{images_dir}'.")

    for fp in files:
        mask, seq, seq_titles = stepOne(fp)
        display_images(seq, seq_titles, f"Step 1: {os.path.basename(fp)}")

        if mask is not None:
            # save the largest CC mask
            name, _  = os.path.splitext(os.path.basename(fp))
            out_name = f"{name}_largestCC.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, mask)
            print(f" → Saved largest CC mask to: {out_path}")

        print("-" * 60)