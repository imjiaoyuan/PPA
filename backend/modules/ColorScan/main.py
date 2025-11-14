import cv2
import numpy as np
import os
import argparse

def get_fruit_color(image_path: str):
    try:
        with open(image_path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError("cv2.imdecode returned None")
    except Exception as e:
        print(f"Error: Could not read or decode image at {image_path}. Reason: {e}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([230, 102, 28])
    upper_bound = np.array([205, 115, 98])
    
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    fruit_pixels = image_rgb[mask > 0]

    if fruit_pixels.size == 0:
        print(f"Warning: No pixels matching the fruit color range were found in {os.path.basename(image_path)}.")
        return None

    average_color = np.mean(fruit_pixels, axis=0)
    average_rgb = tuple(average_color.round().astype(int))
    
    return average_rgb

def process_folder(input_dir: str, output_dir: str):
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_extensions):
            full_path = os.path.join(input_dir, filename)
            print(f"\nProcessing {os.path.normpath(full_path)}...")
            
            avg_color = get_fruit_color(full_path)
            
            if avg_color:
                print(f"Found average fruit RGB color: {avg_color}")
                
                base_filename = os.path.splitext(filename)[0]
                txt_filename = f"{base_filename}.txt"
                txt_output_path = os.path.join(output_dir, txt_filename)
                
                color_string = f"{avg_color[0]}, {avg_color[1]}, {avg_color[2]}"

                try:
                    with open(txt_output_path, 'w', encoding='utf-8') as f:
                        f.write(color_string)
                    print(f"Saved color value to {os.path.normpath(txt_output_path)}")
                except Exception as e:
                    print(f"Error: Could not write to file {txt_output_path}. Reason: {e}")

    print("\nProcessing complete.")

def main():
    parser = argparse.ArgumentParser(description="Calculate average fruit color and save RGB to a text file.")
    parser.add_argument("input_folder", type=str, help="The path to the folder containing input images.")
    parser.add_argument("output_folder", type=str, help="The path to the folder where output .txt files will be saved.")
    
    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()