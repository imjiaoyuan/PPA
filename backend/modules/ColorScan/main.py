import cv2
import numpy as np
import argparse
import os

def get_fruit_color(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image file '{image_path}'")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([0, 100, 100])
    upper_bound = np.array([20, 255, 255])
    
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    fruit_pixels = image_rgb[mask > 0]

    if fruit_pixels.size == 0:
        raise ValueError("No pixels matching the specified color range were found.")

    average_color = np.mean(fruit_pixels, axis=0)
    average_rgb = tuple(average_color.round().astype(int))
    
    return average_rgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average fruit color from an image.")
    parser.add_argument("input_image", type=str, help="The path to the input image.")
    
    args = parser.parse_args()
    
    try:
        avg_color = get_fruit_color(args.input_image)
        print(f"Found average fruit RGB color: {avg_color}")
        color_string = f"{avg_color[0]}, {avg_color[1]}, {avg_color[2]}"
        
        base_filename = os.path.splitext(os.path.basename(args.input_image))[0]
        txt_filename = f"{base_filename}.txt"
        
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(color_string)
        print(f"Saved color value to {txt_filename}")
            
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")