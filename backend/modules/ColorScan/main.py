import cv2
import numpy as np
import argparse
import os

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

def analyze_berry_color(image: np.ndarray, contours: list):
    master_mask = np.zeros(image.shape[:2], dtype="uint8")
    
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
    if not valid_contours:
        raise ValueError("No valid berry contours found after area filtering.")
    cv2.drawContours(master_mask, valid_contours, -1, 255, -1)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    color_filter_mask = cv2.inRange(hsv_image, np.array([0, 80, 50]), np.array([180, 255, 225]))
    
    final_mask = cv2.bitwise_and(master_mask, color_filter_mask)
    
    if cv2.countNonZero(final_mask) == 0:
        raise ValueError("Could not extract valid berry color pixels after filtering.")

    avg_bgr_color = cv2.mean(image, mask=final_mask)[:3]
    avg_bgr_color = np.uint8(avg_bgr_color)
    
    avg_rgb_color = (avg_bgr_color[2], avg_bgr_color[1], avg_bgr_color[0])
    
    return avg_rgb_color

def process_image(input_path: str):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    berry_contours = segment_individual_berries(image.copy())
    if not berry_contours:
        raise ValueError("No berries found in image.")

    avg_rgb = analyze_berry_color(image.copy(), berry_contours)
    
    return avg_rgb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze the average color of berries and save to a text file.')
    parser.add_argument('input_image', type=str, help='Path to the input image.')
    parser.add_argument('output_txt', type=str, help='Path to save the output RGB values (.txt).')
    args = parser.parse_args()

    try:
        rgb_value = process_image(args.input_image)
        
        with open(args.output_txt, 'w') as f:
            f.write(f"{rgb_value[0]},{rgb_value[1]},{rgb_value[2]}")
            
        print(f"Output RGB value saved to: {args.output_txt}")
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")