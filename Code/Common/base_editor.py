# The setup code of the below code need to modified 
# JSON file which is generated is different from which is utilizing in the project. 

import cv2
import json
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class BaseEditor(ABC):
    def __init__(self, window_name: str, image_path: str):
        self.window_name = window_name
        self.image_path = image_path
        
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.orig_h, self.orig_w = self.img.shape[:2]
        self.window_w, self.window_h = self.orig_w, self.orig_h
        
        self.zoom = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 5.0
        self.zoom_step = 0.2
        
        self.view_x = 0.0
        self.view_y = 0.0
        
        self.is_drawing = False
        self.start_x = 0
        self.start_y = 0
        self.last_x = 0
        self.last_y = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self.dragging = False
        
        self.boxes: List[Tuple] = []
        self.current_box: Optional[Tuple] = None
        
        self.colors = [
            (0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)
        ]
        
        self.label_prefix = "B"
    
    def get_color(self, idx: int) -> Tuple:
        return self.colors[idx % len(self.colors)]
    
    def screen_to_image(self, sx: int, sy: int) -> Tuple[int, int]:
        real_x = int(self.view_x + sx / self.zoom)
        real_y = int(self.view_y + sy / self.zoom)
        return real_x, real_y
    
    def image_to_screen(self, ix: int, iy: int) -> Tuple[int, int]:
        sx = int((ix - self.view_x) * self.zoom)
        sy = int((iy - self.view_y) * self.zoom)
        return sx, sy
    
    def clamp_view(self):
        view_w = self.window_w / self.zoom
        view_h = self.window_h / self.zoom
        self.view_x = max(0, min(self.view_x, self.orig_w - view_w))
        self.view_y = max(0, min(self.view_y, self.orig_h - view_h))
    
    def get_box_coords(self, box: Tuple) -> Tuple[int, int, int, int]:
        x1 = min(box[0], box[2])
        y1 = min(box[1], box[3])
        x2 = max(box[0], box[2])
        y2 = max(box[1], box[3])
        return int(x1), int(y1), int(x2), int(y2)
    
    def is_point_in_box(self, px: int, py: int, box: Tuple) -> bool:
        x1, y1, x2, y2 = self.get_box_coords(box)
        return x1 <= px <= x2 and y1 <= py <= y2
    
    def mouse_handler(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y
        
        if event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = self.zoom
            if flags > 0:
                self.zoom = min(self.zoom + self.zoom_step, self.max_zoom)
            else:
                self.zoom = max(self.zoom - self.zoom_step, self.min_zoom)
            
            if old_zoom != self.zoom:
                mx, my = self.screen_to_image(x, y)
                self.view_x = mx - (x / self.zoom)
                self.view_y = my - (y / self.zoom)
                self.clamp_view()
                self.last_x, self.last_y = x, y
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            real_x, real_y = self.screen_to_image(x, y)
            self.start_x, self.start_y = real_x, real_y
            self.is_drawing = True
            self.dragging = True
            self.last_x, self.last_y = x, y
            self.on_draw_start(real_x, real_y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            real_x, real_y = self.screen_to_image(x, y)
            self.current_box = (self.start_x, self.start_y, real_x, real_y)
            self.on_draw_move(real_x, real_y)
        
        elif event == cv2.EVENT_LBUTTONUP and self.is_drawing:
            self.is_drawing = False
            self.dragging = False
            if self.current_box:
                x1, y1, x2, y2 = self.get_box_coords(self.current_box)
                if self.is_valid_box(x1, y1, x2, y2):
                    self.add_box(self.current_box)
                    self.on_box_added(self.current_box)
                else:
                    self.on_invalid_box()
            self.current_box = None
        
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and not self.is_drawing:
            dx = (self.last_x - x) / self.zoom
            dy = (self.last_y - y) / self.zoom
            self.view_x += dx
            self.view_y += dy
            self.clamp_view()
            self.last_x, self.last_y = x, y
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            real_x, real_y = self.screen_to_image(x, y)
            self.on_right_click(real_x, real_y)
    
    def is_valid_box(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        return abs(x2 - x1) > 10 and abs(y2 - y1) > 10
    
    def add_box(self, box: Tuple):
        self.boxes.append(box)
    
    @abstractmethod
    def on_draw_start(self, x: int, y: int):
        pass
    
    @abstractmethod
    def on_draw_move(self, x: int, y: int):
        pass
    
    @abstractmethod
    def on_box_added(self, box: Tuple):
        pass
    
    @abstractmethod
    def on_invalid_box(self):
        pass
    
    def on_right_click(self, x: int, y: int):
        for i, box in enumerate(self.boxes):
            if self.is_point_in_box(x, y, box):
                self.on_delete(i, box)
                self.boxes.pop(i)
                break
    
    def on_delete(self, idx: int, box: Tuple):
        print(f"Deleted: {self.label_prefix}{idx + 1}")
    
    def undo(self):
        if self.boxes:
            removed = self.boxes.pop()
            x1, y1, x2, y2 = self.get_box_coords(removed)
            print(f"Undo: Removed ({x1},{y1})-({x2},{y2})")
    
    def clear_all(self):
        self.boxes.clear()
        print("Cleared all boxes")
    
    def draw_box(self, display, box: Tuple, color: Tuple, label: str, thickness: int = 2):
        x1, y1, x2, y2 = self.get_box_coords(box)
        sx1, sy1 = self.image_to_screen(x1, y1)
        sx2, sy2 = self.image_to_screen(x2, y2)
        cv2.rectangle(display, (sx1, sy1), (sx2, sy2), color, thickness)
        cv2.putText(display, label, (sx1 + 5, sy1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def draw_info_panel(self, display, extra_text: str = ""):
        panel_h = 100
        cv2.rectangle(display, (0, 0), (300, panel_h), (0, 0, 0), -1)
        cv2.putText(display, self.window_name.upper(), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Boxes: {len(self.boxes)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"Zoom: {self.zoom:.1f}x", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        img_x, img_y = self.screen_to_image(self.mouse_x, self.mouse_y)
        cv2.putText(display, f"Pos: ({img_x},{img_y})", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        if extra_text:
            cv2.putText(display, extra_text, (310, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    def render(self, display):
        view_w = int(self.window_w / self.zoom)
        view_h = int(self.window_h / self.zoom)
        
        view_x_c = max(0, min(self.view_x, self.orig_w - view_w))
        view_y_c = max(0, min(self.view_y, self.orig_h - view_h))
        
        roi = self.img[int(view_y_c):int(view_y_c)+view_h, int(view_x_c):int(view_x_c)+view_w]
        if display is None:
            display = np.zeros((int(view_h * self.zoom), int(view_w * self.zoom), 3), dtype=np.uint8)
        else:
            display = cv2.resize(roi, (int(view_w * self.zoom), int(view_h * self.zoom)))
        
        for i, box in enumerate(self.boxes):
            color = self.get_color(i)
            self.draw_box(display, box, color, f"{self.label_prefix}{i+1}")
        
        if self.current_box:
            color = self.get_color(len(self.boxes))
            self.draw_box(display, self.current_box, color, f"{self.label_prefix}{len(self.boxes)+1}", 2)
        
        return display
    
    @abstractmethod
    def get_save_data(self) -> Dict[str, Any]:
        pass
    
    def save(self, filepath: str):
        save_data = self.get_save_data()
        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Saved to: {filepath}")
    
    def run(self, save_path: str):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_w, self.window_h)
        cv2.setMouseCallback(self.window_name, self.mouse_handler)
        
        self.print_instructions()
        
        while True:
            display = self.render(np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8))
            extra = self.get_extra_info()
            self.draw_info_panel(display, extra)
            
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                cv2.destroyAllWindows()
                exit()
            
            elif key == 13 or key == ord('s') or key == ord('S'):
                if len(self.boxes) > 0:
                    self.save(save_path)
                    break
                else:
                    print("Draw at least one box!")
            
            elif key == ord('z') or key == ord('Z'):
                self.undo()
            
            elif key == ord('c') or key == ord('C'):
                self.clear_all()
        
        cv2.destroyAllWindows()
        self.print_summary()
    
    def print_instructions(self):
        print(f"\n{'=' * 50}")
        print(self.window_name.upper())
        print(f"{'=' * 50}")
        print("  LEFT CLICK & DRAG: Draw box")
        print("  RIGHT CLICK: Delete box")
        print("  SCROLL: Zoom (centered on mouse)")
        print("  DRAG (no draw): Pan view")
        print("  Z: Undo last box")
        print("  C: Clear all")
        print("  S or ENTER: Save and exit")
        print("  ESC: Exit without saving")
        print(f"{'=' * 50}")
    
    def print_summary(self):
        print(f"\n{'=' * 50}")
        print(f"{self.window_name.upper()} COMPLETE")
        print(f"{'=' * 50}")
        print(f"Saved {len(self.boxes)} box(es)")
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self.get_box_coords(box)
            print(f"  {self.label_prefix}{i+1}: ({x1},{y1})-({x2},{y2})")
        print(f"{'=' * 50}")
    
    def get_extra_info(self) -> str:
        return ""


class SingleBoxEditor(BaseEditor):
    def on_draw_start(self, x: int, y: int):
        pass
    
    def on_draw_move(self, x: int, y: int):
        pass
    
    def on_box_added(self, box: Tuple):
        x1, y1, x2, y2 = self.get_box_coords(box)
        print(f"{self.label_prefix}{len(self.boxes)}: ({x1},{y1})-({x2},{y2})")
    
    def on_invalid_box(self):
        pass
    
    def get_save_data(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "image_width": self.orig_w,
            "image_height": self.orig_h,
            "boxes": [
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                for x1, y1, x2, y2 in (self.get_box_coords(b) for b in self.boxes)
            ]
        }
