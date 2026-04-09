import cv2
import json
import argparse
import os


def draw_boxes_on_image(image_path: str, json_path: str, output_path: str = None):
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    if not os.path.exists(json_path):
        print(f"Error: JSON not found: {json_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image: {image_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    boxes = []
    box_type = None
    
    if 'terminal_boxes' in data:
        boxes = data['terminal_boxes']
        box_type = 'terminal_boxes'
    elif 'parking_boxes' in data:
        boxes = data['parking_boxes']
        box_type = 'parking_boxes'
    elif 'boxes' in data:
        boxes_data = data['boxes']
        if boxes_data and isinstance(boxes_data[0], dict):
            for b in boxes_data:
                boxes.append([b['x1'], b['y1'], b['x2'], b['y2']])
        else:
            boxes = boxes_data
        box_type = 'boxes'
    else:
        print("No boxes found in JSON")
        return
    
    colors = [
        (0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)
    ]
    
    print(f"\nLoaded {len(boxes)} box(es) from '{box_type}' in {json_path}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        
        if isinstance(box, dict):
            x1, y1 = box.get('x1', 0), box.get('y1', 0)
            x2, y2 = box.get('x2', 0), box.get('y2', 0)
        elif len(box) == 4:
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
        else:
            continue
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        label = f"{i+1}"
        if isinstance(box, dict) and 'x1' in box:
            label = f"{i+1}: ({x1},{y1})-({x2},{y2})"
        else:
            label = f"{i+1}: ({x1},{y1}) w={x2-x1} h={y2-y1}"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        print(f"  Box {i+1}: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")
    
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"\nSaved result to: {output_path}")
    
    cv2.namedWindow('Box Viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Box Viewer', min(1200, img.shape[1]), min(800, img.shape[0]))
    cv2.imshow('Box Viewer', img)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize boxes from JSON on image')
    parser.add_argument('--image', '-i', type=str, required=True, help='Path to image file')
    parser.add_argument('--json', '-j', type=str, required=True, help='Path to JSON file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output image path (optional)')
    
    args = parser.parse_args()
    draw_boxes_on_image(args.image, args.json, args.output)
