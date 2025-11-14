import cv2
import numpy as np
import argparse
import os
import csv

def find_ruler_and_calibrate(image: np.ndarray, ruler_real_length_cm: float = 20.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
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

def segment_fruit_clusters(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    mask = mask1 + mask2
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []

    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) > 2000:
        return [largest_contour]
    else:
        return []

def calculate_and_visualize(image: np.ndarray, contours: list, pixels_per_cm: float):
    all_measurements = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        width_cm = w / pixels_per_cm
        height_cm = h / pixels_per_cm
        
        text_width = f"Width: {width_cm:.2f} cm"
        text_height = f"Height: {height_cm:.2f} cm"

        cv2.putText(image, text_width, (x + 10, y + 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(image, text_height, (x + 10, y + 60), font, 0.8, (255, 255, 255), 2)
        
        tick_interval_px = int(1.0 * pixels_per_cm)
        num_ticks = int(w / tick_interval_px)
        
        for j in range(num_ticks + 1):
            tick_x = x + j * tick_interval_px
            cv2.line(image, (tick_x, y), (tick_x, y - 10), (255, 255, 0), 2)
            cv2.putText(image, str(j), (tick_x - 10, y - 20), font, 0.6, (255, 255, 0), 2)

        all_measurements.append({
            "cluster_id": i + 1,
            "width_cm": round(width_cm, 2),
            "height_cm": round(height_cm, 2)
        })
        
    return image, all_measurements

def save_image_result(output_path: str, image_data: np.ndarray):
    cv2.imwrite(output_path, image_data)
    print(f"Output image saved to: {output_path}")

def save_csv_result(output_path: str, measurement_data: list):
    if not measurement_data:
        return
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = measurement_data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(measurement_data)
    print(f"Measurement data saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Calculate fruit cluster size and output image and CSV data.')
    parser.add_argument('input_image', type=str, help='Path to the input image.')
    parser.add_argument('output_image', type=str, help='Path to save the output image (.jpg, .png).')
    parser.add_argument('output_csv', type=str, help='Path to save the output CSV file (.csv).')
    args = parser.parse_args()

    image = cv2.imread(args.input_image)
    if image is None:
        print(f"Error: Could not read input image: {args.input_image}")
        return

    pixels_per_cm = find_ruler_and_calibrate(image.copy())
    if pixels_per_cm is None:
        print("Error: Ruler not found in image.")
        return

    cluster_contours = segment_fruit_clusters(image.copy())
    if not cluster_contours:
        print("Error: No fruit clusters found in image.")
        return

    result_image, measurements = calculate_and_visualize(image.copy(), cluster_contours, pixels_per_cm)
    
    print("\nAnalysis Results:")
    for meas in measurements:
        print(f"Cluster {meas['cluster_id']}: Width={meas['width_cm']} cm, Height={meas['height_cm']} cm")
    print("")
    
    save_image_result(args.output_image, result_image)
    save_csv_result(args.output_csv, measurements)

if __name__ == '__main__':
    main()