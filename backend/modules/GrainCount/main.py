import cv2
import numpy as np
import argparse
import os

def find_ruler_and_calibrate(image: np.ndarray, ruler_real_length_cm: float = 20.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ruler_candidates = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:
            continue
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 10:
            ruler_candidates.append(rect)

    if not ruler_candidates:
        return None

    main_ruler_rect = max(ruler_candidates, key=lambda r: max(r[1][0], r[1][1]))
    ruler_pixel_width = max(main_ruler_rect[1][0], main_ruler_rect[1][1])
    
    pixels_per_cm = ruler_pixel_width / ruler_real_length_cm
    return pixels_per_cm

def segment_individual_berries(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([25, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    
    lower2 = np.array([170, 100, 100])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)
    
    mask = mask1 + mask2
    
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    sure_fg = cv2.erode(opening, kernel, iterations=3)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    markers = cv2.watershed(image, markers)
    
    berry_contours = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        
        mask_label = np.zeros(markers.shape, dtype="uint8")
        mask_label[markers == label] = 255
        
        contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            berry_contours.append(contours[0])
            
    return berry_contours

def visualize_and_count_berries(image: np.ndarray, contours: list):
    berry_count = 0
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        berry_count += 1
        
    return image, berry_count

def process_image(input_path: str):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    pixels_per_cm = find_ruler_and_calibrate(image.copy())
    if pixels_per_cm is None:
        raise ValueError("Ruler not found in image.")

    berry_contours = segment_individual_berries(image.copy())
    if not berry_contours:
        raise ValueError("No berries found in image.")

    result_image, berry_count = visualize_and_count_berries(image.copy(), berry_contours)
    
    return result_image, berry_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Identify and count all individual berries in an image.')
    parser.add_argument('input_image', type=str, help='Path to the input image.')
    parser.add_argument('output_image', type=str, help='Path to save the output image (.jpg, .png).')
    args = parser.parse_args()

    try:
        res_img, count = process_image(args.input_image)
        
        h, w, _ = res_img.shape
        text = f"{count} berries"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_color = (255, 255, 255)
        thickness = 3
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = w - text_size[0] - 30
        text_y = h - 30
        
        cv2.putText(res_img, text, (text_x, text_y), font, font_scale, font_color, thickness)
        
        cv2.imwrite(args.output_image, res_img)
        print(f"Output image saved to: {args.output_image}")
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")