import cv2
import numpy as np
import os

def classify_usb(contour, image):
    """Comprehensive USB port classifier with 5 types"""
    # Get geometric features
    rect = cv2.minAreaRect(contour)
    (_, _), (width, height), angle = rect
    aspect_ratio = max(width, height) / min(width, height)
    
    # Calculate solidity and convexity defects
    hull = cv2.convexHull(contour)
    solidity = cv2.contourArea(contour) / cv2.contourArea(hull)
    
    # Ellipse fitting for ovalness
    ellipse = cv2.fitEllipse(contour)
    (_, _), (ma, MA), _ = ellipse
    ellipse_ratio = MA / ma
    
    # Width ratio (top/bottom)
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    width_ratio = (rightmost[0]-leftmost[0]) / (bottommost[0]-topmost[0]) if (bottommost[0]-topmost[0]) != 0 else 1
    
    # Strict classification rules
    if 3.0 <= aspect_ratio <= 4.0 and 0.9 <= ellipse_ratio <= 1.1:
        return "USB-C (Oval)"
    elif 2.1 <= aspect_ratio <= 2.8 and width_ratio < 1.1:
        return "USB-A (Rectangle)"
    elif 1.0 <= aspect_ratio <= 1.2 and solidity > 0.95:
        return "USB-B (Square)"
    elif 1.5 <= aspect_ratio <= 2.0 and width_ratio > 1.2:
        return "Micro-USB (Trapezoid)"
    elif 1.3 <= aspect_ratio <= 1.7 and 1.0 <= width_ratio <= 1.2:
        return "Mini-USB (Beveled)"
    else:
        return f"Unknown (AR:{aspect_ratio:.1f}, WR:{width_ratio:.1f}, ER:{ellipse_ratio:.1f})"

def process_image(image_path):
    """Enhanced processing pipeline"""
    image = cv2.imread(image_path)
    if image is None:
        return None, "Invalid image"
    
    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9,9), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Morphology
    kernel = np.ones((7,7), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "No contours found"
    
    contour = max(contours, key=cv2.contourArea)
    usb_type = classify_usb(contour, image)
    
    # Visualization
    result_img = image.copy()
    cv2.drawContours(result_img, [contour], -1, (0,255,0), 2)
    
    # Draw both rectangle and ellipse
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    cv2.drawContours(result_img, [np.int32(box)], 0, (255,0,0), 2)
    cv2.ellipse(result_img, cv2.fitEllipse(contour), (0,0,255), 2)
    
    # Add classification text
    cv2.putText(result_img, usb_type, (10,30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    return result_img, usb_type

def process_folder(folder_path):
    """Process all images in folder"""
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    valid_ext = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        print(f"\nProcessing: {img_file}")
        
        try:
            result_img, usb_type = process_image(img_path)
            if result_img is not None:
                cv2.imshow(f"Result: {usb_type}", result_img)
                print(f"Classification: {usb_type},")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

if __name__ == "__main__":
    folder_path = input("Enter folder path containing USB images: ").strip()
    process_folder(folder_path)