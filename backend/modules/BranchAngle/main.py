import cv2
import numpy as np
import math
import sys

def find_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None
    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = t_num / den
    return (int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1)))

def calculate_acute_angle_from_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    cosine_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosine_angle))
    return angle_deg

def process_image(input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read image file '{input_path}'")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=80, maxLineGap=20)
    if lines is None or len(lines) < 2:
        print("Error: Not enough lines were detected.")
        return

    vertical_lines = []
    diagonal_lines = []
    for line_segment in lines:
        line = line_segment[0]
        angle = math.degrees(math.atan2(line[3] - line[1], line[2] - line[0]))
        length = np.linalg.norm((line[2] - line[0], line[3] - line[1]))
        if 80 < abs(angle) < 100:
            vertical_lines.append((line, length))
        elif 10 < abs(angle) < 70 or 110 < abs(angle) < 170:
            diagonal_lines.append((line, length))

    if not vertical_lines or not diagonal_lines:
        print("Error: Could not find both a trunk and a branch.")
        return

    vertical_lines.sort(key=lambda x: x[1], reverse=True)
    diagonal_lines.sort(key=lambda x: x[1], reverse=True)
    trunk_line = vertical_lines[0][0]
    branch_line = diagonal_lines[0][0]

    intersection = find_intersection(trunk_line, branch_line)
    if intersection is None:
        print("Error: Lines are parallel, cannot find intersection.")
        return

    p_trunk_up = (trunk_line[0], trunk_line[1]) if trunk_line[1] < trunk_line[3] else (trunk_line[2], trunk_line[3])
    p_branch_up = (branch_line[0], branch_line[1]) if branch_line[1] < branch_line[3] else (branch_line[2], branch_line[3])

    vec_trunk = (p_trunk_up[0] - intersection[0], p_trunk_up[1] - intersection[1])
    vec_branch = (p_branch_up[0] - intersection[0], p_branch_up[1] - intersection[1])

    final_angle_value = calculate_acute_angle_from_vectors(vec_trunk, vec_branch)

    cv2.line(image, intersection, p_trunk_up, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(image, intersection, p_branch_up, (0, 255, 0), 3, cv2.LINE_AA)
    
    angle_trunk = np.degrees(np.arctan2(vec_trunk[1], vec_trunk[0]))
    angle_branch = np.degrees(np.arctan2(vec_branch[1], vec_branch[0]))
    
    sweep_forward = (angle_branch - angle_trunk + 360) % 360
    sweep_backward = (angle_trunk - angle_branch + 360) % 360
    
    if abs(sweep_forward - final_angle_value) < abs(sweep_backward - final_angle_value):
        start_angle = angle_trunk
        end_angle = angle_branch
    else:
        start_angle = angle_branch
        end_angle = angle_trunk
        
    overlay = image.copy()
    arc_radius = 80 * 3
    cv2.ellipse(overlay, intersection, (arc_radius, arc_radius), 0, start_angle, end_angle, (255, 200, 0), -1)
    alpha = 0.4
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    if (end_angle - start_angle + 360) % 360 > 180:
         mean_angle_deg = (start_angle + end_angle - 360) / 2.0
    else:
         mean_angle_deg = (start_angle + end_angle) / 2.0
         
    mean_angle_rad = np.deg2rad(mean_angle_deg)
    text_radius = int(arc_radius * 1.05)  
    text_x = int(intersection[0] + text_radius * np.cos(mean_angle_rad))
    text_y = int(intersection[1] + text_radius * np.sin(mean_angle_rad))

    text_combined = f"{final_angle_value:.1f} deg"
    white = (255, 255, 255)
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_font = ImageFont.load_default()
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        tw, th = draw.textsize(text_combined, font=pil_font)
        top_left = (int(text_x - tw / 2), int(text_y - th / 2))
        draw.text(top_left, text_combined, font=pil_font, fill=white)
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, arc_radius / 200.0)
        font_thickness = max(1, int(font_scale * 2))
        (w, h), baseline = cv2.getTextSize(text_combined, font, font_scale, font_thickness)
        origin = (text_x - w // 2, text_y + h // 2)
        cv2.putText(image, text_combined, origin, font, font_scale, white, font_thickness, cv2.LINE_AA)
    
    cv2.imwrite(output_path, image)
    print(f"Processing complete. Result saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python measure_angle.py <input_image_path> <output_image_path>")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    process_image(input_image_path, output_image_path)