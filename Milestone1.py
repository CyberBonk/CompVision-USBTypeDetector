import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# -------------------------------------------------------------------
# CONFIGURATION
BASE_DIR        = os.path.abspath(os.path.dirname(__file__))
IMAGES_DIR      = os.path.join(BASE_DIR, "images")
OUTPUT_DIR      = os.path.join(BASE_DIR, "output_images")

THIN_EDGE_MAX   = 4
MEDIUM_EDGE_MAX = 6
MORPH_PARAMS = {
    "thin":   {"kopen":3,  "iopen":1, "kclose":5,  "iclose":1},
    "medium": {"kopen":5,  "iopen":1, "kclose":9,  "iclose":2},
    "thick":  {"kopen":5,  "iopen":1, "kclose":13, "iclose":2},
}

# -------------------------------------------------------------------
def ensure_dirs():
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Input folder not found: {IMAGES_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------
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
            plt.imshow(np.zeros((50,50), np.uint8), cmap='gray')
        elif img.ndim == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(ttl, fontsize=9)
        plt.xticks([]); plt.yticks([])
    plt.tight_layout(rect=[0,0.02,1,0.94])
    plt.show()

# -------------------------------------------------------------------
def estimate_edge_thickness(edges):
    if edges is None or cv2.countNonZero(edges)==0:
        return 1
    inv = cv2.bitwise_not(edges)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    vals = dist[edges>0]
    if vals.size==0:
        vals = dist[dist>0]
    if vals.size==0:
        return 1
    thickness = (vals*2).astype(int)
    lo, hi = np.percentile(thickness, [5,95])
    inliers = thickness[(thickness>=lo)&(thickness<=hi)]
    med = int(np.median(inliers)) if inliers.size else int(np.median(thickness))
    return max(1, med)

# -------------------------------------------------------------------
def get_rotation_angle_and_deskew(gray,
    angle_tol=25, canny1=50, canny2=150,
    hough_thresh=40, min_len=30, max_gap=10):
    seq, titles = [], []
    h, w = gray.shape[:2]

    # blur + Canny
    b3 = cv2.GaussianBlur(gray, (3,3), 0)
    seq.append(b3.copy()); titles.append("Blur 3×3")
    e = cv2.Canny(b3, canny1, canny2)
    seq.append(e.copy()); titles.append(f"Canny {canny1}-{canny2}")

    # HoughLinesP → collect angles
    lines = cv2.HoughLinesP(e,1,np.pi/180,hough_thresh,
                            minLineLength=min_len,maxLineGap=max_gap)
    viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    angles = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            dx,dy = x2-x1, y2-y1
            if dx==0: continue
            ang = math.degrees(math.atan2(dy,dx))
            if abs(ang)>(180-angle_tol):
                ang -= math.copysign(180, ang)
            color = (0,255,0) if abs(ang)<=angle_tol else (0,0,255)
            cv2.line(viz,(x1,y1),(x2,y2),color,1)
            if abs(ang)<=angle_tol:
                angles.append(ang)
    seq.append(viz.copy()); titles.append("Hough Viz")

    # rotate if needed
    deskewed = gray.copy()
    if angles:
        avg = float(np.mean(angles))
        if abs(avg)>0.5:
            M = cv2.getRotationMatrix2D((w//2,h//2), avg, 1.0)
            deskewed = cv2.warpAffine(gray, M, (w,h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    seq.append(deskewed.copy()); titles.append("Deskewed")
    return deskewed, seq, titles

# -------------------------------------------------------------------
def detect_and_clean_edges(gray):
    b5 = cv2.GaussianBlur(gray, (5,5), 0)
    e5 = cv2.Canny(b5, 40, 100)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    thick = cv2.dilate(e5, k3, iterations=1)

    med = estimate_edge_thickness(thick)
    if med<=THIN_EDGE_MAX:
        p = MORPH_PARAMS["thin"]
    elif med<=MEDIUM_EDGE_MAX:
        p = MORPH_PARAMS["medium"]
    else:
        p = MORPH_PARAMS["thick"]

    ko = cv2.getStructuringElement(cv2.MORPH_RECT,(p["kopen"],p["kopen"]))
    opened = cv2.morphologyEx(thick,cv2.MORPH_OPEN,ko,iterations=p["iopen"])
    kc = cv2.getStructuringElement(cv2.MORPH_RECT,(p["kclose"],p["kclose"]))
    cleaned = cv2.morphologyEx(opened,cv2.MORPH_CLOSE,kc,iterations=p["iclose"])
    return cleaned

# -------------------------------------------------------------------
def extract_largest_cc(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n<=1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best  = int(np.argmax(areas)) + 1
    return (labels==best).astype(np.uint8)*255

# -------------------------------------------------------------------
def make_tight_mask(binary_mask):
    cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tight = np.zeros_like(binary_mask)
    if cnts:
        best = max(cnts, key=lambda c: cv2.contourArea(c))
        cv2.drawContours(tight, [best], -1, 255, thickness=cv2.FILLED)
    return tight

# -------------------------------------------------------------------
def process_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"⚠ Could not load: {path}")
        return None
    seq, titles = [], []

    seq.append(img.copy()); titles.append("Original BGR")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    seq.append(gray.copy()); titles.append("Grayscale")

    gray_ds, ds_seq, ds_titles = get_rotation_angle_and_deskew(gray)
    seq.extend(ds_seq); titles.extend(ds_titles)

    cleaned = detect_and_clean_edges(gray_ds)
    seq.append(cleaned.copy()); titles.append("Cleaned Mask")

    cc_mask = extract_largest_cc(cleaned)
    seq.append(cc_mask.copy()); titles.append("Largest CC")

    tight = make_tight_mask(cc_mask)
    seq.append(tight.copy()); titles.append("Final Mask")

    masked = cv2.bitwise_and(img, img, mask=tight)
    seq.append(masked.copy()); titles.append("Masked Original")

    return {"name": os.path.splitext(os.path.basename(path))[0],
            "mask": tight, "masked": masked,
            "seq": seq, "titles": titles}

# -------------------------------------------------------------------
def main():
    ensure_dirs()
    files = [f for f in sorted(os.listdir(IMAGES_DIR)) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp"))]
    if not files:
        print("No images in", IMAGES_DIR)
        return

    for fn in files:
        path = os.path.join(IMAGES_DIR, fn)
        res = process_image(path)
        if res is None:
            continue
        display_images(res["seq"], res["titles"], res["name"])
        mask_path = os.path.join(OUTPUT_DIR, f"{res['name']}_mask.png")
        img_path  = os.path.join(OUTPUT_DIR, f"{res['name']}_masked.png")
        cv2.imwrite(mask_path, res["mask"])
        cv2.imwrite(img_path,  res["masked"])
        print(f"→ Saved: mask={mask_path}, masked={img_path}")
        print("-"*60)

if __name__ == "__main__":
    main()
