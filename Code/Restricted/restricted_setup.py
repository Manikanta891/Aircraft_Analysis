import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(script_dir, "..")))

import cv2
import json
import numpy as np
import argparse
from typing import Dict, Any, Tuple, Optional
from Common.base_editor import BaseEditor
media_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "Media"))
default_image = os.path.join(media_dir, "restricted.jpg")
default_save = os.path.join(script_dir, "restricted_zones.json")

class RestrictedEditor(BaseEditor):
    def __init__(self, image_path: str):
        super().__init__("Restricted Zone Setup", image_path)
        self.label_prefix = "Zone"
        self.boxes = []
        self.zones = []
        self.mode = "inner"
        self.current_inner = None
        self.current_outer = None
        self.drawing = False
        self.start_x, self.start_y = 0, 0
        self.last_x, self.last_y = 0, 0
    
    def on_draw_start(self, x: int, y: int):
        pass
    
    def on_draw_move(self, x: int, y: int):
        pass
    
    def on_box_added(self, box: Tuple):
        pass
    
    def on_invalid_box(self):
        pass
    
    def get_box_coords(self, box: Tuple) -> Tuple[int, int, int, int]:
        x1 = min(box[0], box[2])
        y1 = min(box[1], box[3])
        x2 = max(box[0], box[2])
        y2 = max(box[1], box[3])
        return int(x1), int(y1), int(x2), int(y2)
    
    def boxes_overlap(self, inner: Tuple, outer: Tuple) -> bool:
        x1i, y1i, x2i, y2i = self.get_box_coords(inner)
        x1o, y1o, x2o, y2o = self.get_box_coords(outer)
        return not (x2i < x1o or x2o < x1i or y2i < y1o or y2o < y1i)
    
    def mouse_handler(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y
        
        if event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = self.zoom
            if flags > 0:
                self.zoom = min(self.zoom + 0.2, self.max_zoom)
            else:
                self.zoom = max(self.zoom - 0.2, self.min_zoom)
            
            if old_zoom != self.zoom:
                mx, my = self.screen_to_image(x, y)
                self.view_x = mx - (x / self.zoom)
                self.view_y = my - (y / self.zoom)
                self.clamp_view()
                self.last_x, self.last_y = x, y
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            real_x, real_y = self.screen_to_image(x, y)
            self.start_x, self.start_y = real_x, real_y
            self.drawing = True
            self.last_x, self.last_y = x, y
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            real_x, real_y = self.screen_to_image(x, y)
            if self.mode == "inner":
                self.current_inner = (self.start_x, self.start_y, real_x, real_y)
            else:
                self.current_outer = (self.start_x, self.start_y, real_x, real_y)
        
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            
            if self.mode == "inner" and self.current_inner:
                x1, y1, x2, y2 = self.get_box_coords(self.current_inner)
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.mode = "outer"
                    self.current_outer = None
                    print(f"Inner zone drawn. Now draw OUTER warning zone...")
                else:
                    self.current_inner = None
            
            elif self.mode == "outer" and self.current_outer:
                x1, y1, x2, y2 = self.get_box_coords(self.current_outer)
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    if self.current_inner and self.boxes_overlap(self.current_inner, self.current_outer):
                        inner_area = abs(self.current_inner[2] - self.current_inner[0]) * abs(self.current_inner[3] - self.current_inner[1])
                        outer_area = abs(self.current_outer[2] - self.current_outer[0]) * abs(self.current_outer[3] - self.current_outer[1])
                        
                        if inner_area < outer_area:
                            xi1, yi1, xi2, yi2 = self.get_box_coords(self.current_inner)
                            xo1, yo1, xo2, yo2 = self.get_box_coords(self.current_outer)
                            
                            zone = {
                                "id": len(self.zones) + 1,
                                "name": f"Zone {len(self.zones) + 1}",
                                "restricted": {"x1": xi1, "y1": yi1, "x2": xi2, "y2": yi2},
                                "warning": {"x1": xo1, "y1": yo1, "x2": xo2, "y2": yo2}
                            }
                            self.zones.append(zone)
                            print(f"Zone {len(self.zones)}: Restricted ({xi1},{yi1})-({xi2},{yi2}), Warning ({xo1},{yo1})-({xo2},{yo2})")
                            self.current_inner = None
                            self.current_outer = None
                            self.mode = "inner"
                        else:
                            print("Error: Inner zone must be smaller than outer zone! Redraw outer box.")
                            self.current_outer = None
                    else:
                        print("Error: Boxes must overlap! Redraw outer box.")
                        self.current_outer = None
                else:
                    self.current_outer = None
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            real_x, real_y = self.screen_to_image(x, y)
            for i, zone in enumerate(self.zones):
                r = zone['restricted']
                if r['x1'] <= real_x <= r['x2'] and r['y1'] <= real_y <= r['y2']:
                    print(f"Deleted: {zone['name']}")
                    self.zones.pop(i)
                    break
    
    def undo(self):
        if self.zones:
            removed = self.zones.pop()
            print(f"Undo: Removed {removed['name']}")
        elif self.mode == "outer":
            self.mode = "inner"
            self.current_inner = None
            print("Undo: Cleared inner zone")
    
    def clear_all(self):
        self.zones.clear()
        self.mode = "inner"
        self.current_inner = None
        self.current_outer = None
        print("Cleared all zones")
    
    def render(self, display):
        view_w = int(self.window_w / self.zoom)
        view_h = int(self.window_h / self.zoom)
        
        view_x_c = max(0, min(self.view_x, self.orig_w - view_w))
        view_y_c = max(0, min(self.view_y, self.orig_h - view_h))
        
        img = self.img
        roi = img[int(view_y_c):int(view_y_c)+view_h, int(view_x_c):int(view_x_c)+view_w]
        display = cv2.resize(roi, (int(view_w * self.zoom), int(view_h * self.zoom)))
        
        for i, zone in enumerate(self.zones):
            color = self.get_color(i)
            r = zone['restricted']
            w = zone['warning']
            
            ri_x1, ri_y1 = self.image_to_screen(r['x1'], r['y1'])
            ri_x2, ri_y2 = self.image_to_screen(r['x2'], r['y2'])
            
            overlay = display.copy()
            cv2.rectangle(overlay, (ri_x1, ri_y1), (ri_x2, ri_y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            cv2.rectangle(display, (ri_x1, ri_y1), (ri_x2, ri_y2), (0, 0, 255), 3)
            
            wi_x1, wi_y1 = self.image_to_screen(w['x1'], w['y1'])
            wi_x2, wi_y2 = self.image_to_screen(w['x2'], w['y2'])
            
            overlay = display.copy()
            cv2.rectangle(overlay, (wi_x1, wi_y1), (wi_x2, wi_y2), (0, 165, 255), -1)
            cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
            cv2.rectangle(display, (wi_x1, wi_y1), (wi_x2, wi_y2), (0, 165, 255), 2)
            
            cx = (ri_x1 + ri_x2) // 2
            cy = ri_y1 - 10
            cv2.putText(display, zone['name'], (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if self.current_inner:
            x1, y1, x2, y2 = self.get_box_coords(self.current_inner)
            sx1, sy1 = self.image_to_screen(x1, y1)
            sx2, sy2 = self.image_to_screen(x2, y2)
            cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
            cv2.putText(display, "INNER (Restricted)", (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        if self.current_outer:
            x1, y1, x2, y2 = self.get_box_coords(self.current_outer)
            sx1, sy1 = self.image_to_screen(x1, y1)
            sx2, sy2 = self.image_to_screen(x2, y2)
            cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 165, 255), 2)
            cv2.putText(display, "OUTER (Warning)", (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
        return display
    
    def draw_info_panel(self, display, extra_text: str = ""):
        panel_h = 120
        panel = np.zeros((panel_h, 350, 3), dtype=np.uint8)
        display[0:panel_h, 0:350] = panel
        
        cv2.putText(display, "RESTRICTED ZONE SETUP", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(display, f"Zones: {len(self.zones)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"Zoom: {self.zoom:.1f}x", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        mode_text = f"Mode: {self.mode.upper()}" if not (self.current_inner or self.current_outer) else "Draw box..."
        cv2.putText(display, mode_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        img_x, img_y = self.screen_to_image(self.mouse_x, self.mouse_y)
        cv2.putText(display, f"Pos: ({img_x},{img_y})", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def get_save_data(self) -> Any:
        return self.zones
    
    def print_instructions(self):
        print(f"\n{'=' * 60}")
        print("RESTRICTED ZONE SETUP (Dual Zone System)")
        print(f"{'=' * 60}")
        print("  1. Draw INNER box (restricted area)")
        print("  2. Draw OUTER box (warning area)")
        print("  3. Repeat for more zones")
        print()
        print("  LEFT CLICK & DRAG: Draw zone")
        print("  RIGHT CLICK: Delete zone")
        print("  SCROLL: Zoom (centered on mouse)")
        print("  DRAG (no draw): Pan view")
        print("  Z: Undo last zone / Clear current")
        print("  C: Clear all")
        print("  S or ENTER: Save and exit")
        print("  ESC: Exit without saving")
        print(f"{'=' * 60}")
    
    def print_summary(self):
        print(f"\n{'=' * 60}")
        print("RESTRICTED ZONE SETUP COMPLETE")
        print(f"{'=' * 60}")
        for zone in self.zones:
            r = zone['restricted']
            w = zone['warning']
            print(f"  {zone['name']}:")
            print(f"    Restricted: ({r['x1']},{r['y1']})-({r['x2']},{r['y2']})")
            print(f"    Warning: ({w['x1']},{w['y1']})-({w['x2']},{w['y2']})")
        print(f"{'=' * 60}")
    
    def run(self, save_path: str = None):
        output_path = save_path if save_path else default_save
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_w, self.window_h)
        cv2.setMouseCallback(self.window_name, self.mouse_handler)
        
        self.print_instructions()
        
        while True:
            display = self.render(None)
            self.draw_info_panel(display)
            
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                cv2.destroyAllWindows()
                exit()
            
            elif key == 13 or key == ord('s') or key == ord('S'):
                if len(self.zones) > 0:
                    with open(output_path, 'w') as f:
                        json.dump(self.zones, f, indent=2)
                    print(f"Saved {len(self.zones)} zone(s) to: {output_path}")
                    break
                else:
                    print("Draw at least one restricted zone!")
            
            elif key == ord('z') or key == ord('Z'):
                self.undo()
            
            elif key == ord('c') or key == ord('C'):
                self.clear_all()
        
        cv2.destroyAllWindows()
        self.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Restricted Zone Setup')
    parser.add_argument('--image', type=str, default=None, help='Path to image file')
    parser.add_argument('--load', type=str, default=None, help='Load existing JSON file')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    args = parser.parse_args()

    img_path = args.image if args.image else default_image
    if not os.path.exists(img_path):
        print(f"Error: Image not found: {img_path}")
        exit()

    editor = RestrictedEditor(img_path)

    if args.load and os.path.exists(args.load):
        with open(args.load, 'r') as f:
            editor.zones = json.load(f)
        print(f"Loaded {len(editor.zones)} zone(s) from {args.load}")

    save_path = args.output if args.output else default_save
    editor.run(save_path)
