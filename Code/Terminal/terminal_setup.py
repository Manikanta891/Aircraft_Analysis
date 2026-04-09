import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(script_dir, "..")))

import json
import argparse
from Common.base_editor import SingleBoxEditor

ROOT_DIR = os.path.normpath(os.path.join(script_dir, "..", ".."))
media_dir = os.path.join(ROOT_DIR, "Media")
default_image = os.path.join(media_dir, "terminal.jpg")
default_save = os.path.join(ROOT_DIR, "Airport_Simulator", "terminals.json")

class TerminalEditor(SingleBoxEditor):
    def __init__(self, image_path: str):
        super().__init__("Terminal Setup", image_path)
        self.label_prefix = "T"
        self.default_save = default_save

    def get_save_path(self) -> str:
        return self.default_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Terminal Setup')
    parser.add_argument('--image', type=str, default=None, help='Path to image file')
    parser.add_argument('--load', type=str, default=None, help='Load existing JSON file')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    args = parser.parse_args()

    img_path = args.image if args.image else default_image
    if not os.path.exists(img_path):
        print(f"Error: Image not found: {img_path}")
        exit()

    editor = TerminalEditor(img_path)

    if args.load and os.path.exists(args.load):
        with open(args.load, 'r') as f:
            data = json.load(f)
            if 'terminal_boxes' in data:
                editor.boxes = [(b['x1'], b['y1'], b['x2'], b['y2']) for b in data['terminal_boxes']]
            elif 'boxes' in data:
                editor.boxes = [(b['x1'], b['y1'], b['x2'], b['y2']) for b in data['boxes']]
        print(f"Loaded {len(editor.boxes)} box(es) from {args.load}")

    save_path = args.output if args.output else default_save
    editor.default_save = save_path
    editor.run(save_path)
