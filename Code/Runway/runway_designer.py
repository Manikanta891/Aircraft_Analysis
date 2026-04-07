import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(script_dir, "..")))

import json
import argparse
from typing import Tuple, Dict, Any, Optional
import cv2
from Common.base_editor import BaseEditor
ROOT_DIR = os.path.normpath(os.path.join(script_dir, "..", ".."))
media_dir = os.path.join(ROOT_DIR, "Media")
default_image = os.path.join(media_dir, "runway.png")
save_path = os.path.join(script_dir, "runways.json")

class RunwayEditor(BaseEditor):
    def __init__(self, image_path: str):
        super().__init__("Runway Designer", image_path)
        self.label_prefix = "R"
    
    def on_draw_start(self, x: int, y: int):
        pass
    
    def on_draw_move(self, x: int, y: int):
        pass
    
    def on_box_added(self, box: Tuple):
        x1, y1, x2, y2 = self.get_box_coords(box)
        print(f"Runway {len(self.boxes)}: ({x1},{y1})-({x2},{y2})")
    
    def on_invalid_box(self):
        pass
    
    def get_save_data(self) -> Any:
        return [
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            for x1, y1, x2, y2 in (self.get_box_coords(b) for b in self.boxes)
        ]
    
    def run(self, output_path: str, load_path: Optional[str] = None):
        if load_path and os.path.exists(load_path):
            with open(load_path, 'r') as f:
                loaded = json.load(f)
                self.boxes.extend([(b['x1'], b['y1'], b['x2'], b['y2']) for b in loaded])
            print(f"Loaded {len(loaded)} runway(s) from {load_path}")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_w, self.window_h)
        cv2.setMouseCallback(self.window_name, self.mouse_handler)
        
        self.print_instructions()
        
        while True:
            display = self.render(None)
            extra = self.get_extra_info()
            self.draw_info_panel(display, extra)
            
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                cv2.destroyAllWindows()
                exit()
            
            elif key == 13 or key == ord('s') or key == ord('S'):
                if len(self.boxes) > 0:
                    with open(output_path, 'w') as f:
                        json.dump(self.get_save_data(), f, indent=4)
                    print(f"Saved {len(self.boxes)} runway(s) to {output_path}")
                    break
                else:
                    print("Draw at least one runway!")
            
            elif key == ord('z') or key == ord('Z'):
                self.undo()
            
            elif key == ord('c') or key == ord('C'):
                self.clear_all()
        
        cv2.destroyAllWindows()
        self.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runway Designer')
    parser.add_argument('--image', type=str, default=None, help='Path to image file')
    parser.add_argument('--video', type=str, default=None, help='Path to video file (uses first frame)')
    parser.add_argument('--frame', type=int, default=0, help='Frame number for video')
    parser.add_argument('--load', type=str, default=None, help='Load existing JSON file')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    args = parser.parse_args()

    if args.video and os.path.exists(args.video):
        import numpy as np
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, frame = cap.read()
        cap.release()
        if ret:
            img_path = args.video
        else:
            print(f"Error: Cannot read video: {args.video}")
            exit()
    else:
        img_path = args.image if args.image and os.path.exists(args.image) else default_image

    if not os.path.exists(img_path):
        print(f"Error: Image not found: {img_path}")
        exit()

    editor = RunwayEditor(img_path)
    output = args.output if args.output else save_path
    editor.run(output, args.load)
