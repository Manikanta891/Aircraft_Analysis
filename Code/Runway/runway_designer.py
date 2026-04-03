import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(script_dir).resolve().parent
DEFAULT_IMAGE_PATH = ROOT_DIR / "Media" / "runway.png"
DEFAULT_OUTPUT_PATH = os.path.join(script_dir, "runways.json")

drawing = False
start_point = None
end_point = None
runways = []
current_image = None
image_copy = None


def load_runways_from_file(filepath: str) -> list:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return []


def save_runways_to_file(runways: list, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(runways, f, indent=4)
    print(f"Runways saved to: {filepath}")


def draw_runways(img: np.ndarray, runways_list: list, highlight_idx: int = -1):
    for i, runway in enumerate(runways_list):
        x1, y1 = int(runway['x1']), int(runway['y1'])
        x2, y2 = int(runway['x2']), int(runway['y2'])

        if i == highlight_idx:
            color = (0, 255, 255)
            thickness = 3
        else:
            color = (0, 165, 255)
            thickness = 2

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = f"Runway {i + 1}: ({x1},{y1})-({x2},{y2})"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_preview(img: np.ndarray):
    global start_point, end_point, drawing
    if drawing and start_point and end_point:
        x1 = min(start_point[0], end_point[0])
        y1 = min(start_point[1], end_point[1])
        x2 = max(start_point[0], end_point[0])
        y2 = max(start_point[1], end_point[1])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Preview: ({x1},{y1})-({x2},{y2})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, end_point, runways, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and current_image is not None:
            end_point = (x, y)
            image_copy = current_image.copy()
            draw_runways(image_copy, runways)
            draw_preview(image_copy)
            cv2.imshow("Runway Designer", image_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

        if start_point and end_point:
            x1 = min(start_point[0], end_point[0])
            y1 = min(start_point[1], end_point[1])
            x2 = max(start_point[0], end_point[0])
            y2 = max(start_point[1], end_point[1])

            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                runway = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                runways.append(runway)
                print(f"Added Runway {len(runways)}: ({x1},{y1})-({x2},{y2})")

                if current_image is not None:
                    image_copy = current_image.copy()
                    draw_runways(image_copy, runways)
                    cv2.imshow("Runway Designer", image_copy)

        start_point = None
        end_point = None


def is_point_in_runway(px: float, py: float, runway: dict) -> bool:
    x1 = min(runway['x1'], runway['x2'])
    x2 = max(runway['x1'], runway['x2'])
    y1 = min(runway['y1'], runway['y2'])
    y2 = max(runway['y1'], runway['y2'])
    return x1 <= px <= x2 and y1 <= py <= y2


def is_aircraft_on_any_runway(cx: float, cy: float, runways_list: list) -> bool:
    for runway in runways_list:
        if is_point_in_runway(cx, cy, runway):
            return True
    return False


def load_video_frame(video_path: str, frame_num: int = 0) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame
    return None


def main():
    global current_image, image_copy, runways

    import argparse
    parser = argparse.ArgumentParser(description='Runway Designer - Draw runways on an image')
    parser.add_argument('--image', type=str, help='Path to image file', default=None)
    parser.add_argument('--video', type=str, help='Path to video file (uses first frame)', default=None)
    parser.add_argument('--frame', type=int, help='Frame number for video', default=0)
    parser.add_argument('--output', type=str, help='Output JSON file path', default=None)
    parser.add_argument('--load', type=str, help='Load existing runways from JSON', default=None)

    args = parser.parse_args()

    if args.load:
        runways = load_runways_from_file(args.load)
        print(f"Loaded {len(runways)} runways from: {args.load}")

    if args.image:
        if os.path.exists(args.image):
            current_image = cv2.imread(args.image)
        else:
            print(f"Error: Image not found: {args.image}")
            return
    elif args.video:
        if os.path.exists(args.video):
            current_image = load_video_frame(args.video, args.frame)
            if current_image is None:
                return
        else:
            print(f"Error: Video not found: {args.video}")
            return
    else:
        if os.path.exists(str(DEFAULT_IMAGE_PATH)):
            current_image = load_video_frame(str(DEFAULT_IMAGE_PATH), 0)
            if current_image is None:
                current_image = np.ones((720, 1280, 3), dtype=np.uint8) * 50
                cv2.putText(current_image, "No image loaded - Draw runways here",
                            (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            current_image = np.ones((720, 1280, 3), dtype=np.uint8) * 50
            cv2.putText(current_image, "No image loaded - Draw runways here",
                        (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if current_image is None:
        print("Error: No image available")
        return

    image_copy = current_image.copy()

    cv2.namedWindow("Runway Designer")
    cv2.setMouseCallback("Runway Designer", mouse_callback)

    draw_runways(image_copy, runways)
    cv2.imshow("Runway Designer", image_copy)

    print("\n" + "=" * 60)
    print("RUNWAY DESIGNER - INSTRUCTIONS")
    print("=" * 60)
    print("1. Click and drag to draw a rectangular runway")
    print("2. Repeat to add multiple runways")
    print("3. Press 'z' to undo last runway")
    print("4. Press 'c' to clear all runways")
    print("5. Press 's' to save runways to JSON")
    print("6. Press 'r' to reset image view")
    print("7. Press 'ESC' to exit")
    print("=" * 60)
    print(f"\nCurrent runways: {len(runways)}")
    print("=" * 60)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == ord('s') or key == ord('S'):
            output_path = args.output if args.output else DEFAULT_OUTPUT_PATH
            save_runways_to_file(runways, output_path)

        elif key == ord('z') or key == ord('Z'):
            if runways:
                removed = runways.pop()
                print(f"Removed last runway: {removed}")
                image_copy = current_image.copy()
                draw_runways(image_copy, runways)
                cv2.imshow("Runway Designer", image_copy)

        elif key == ord('c') or key == ord('C'):
            runways.clear()
            print("Cleared all runways")
            image_copy = current_image.copy()
            draw_runways(image_copy, runways)
            cv2.imshow("Runway Designer", image_copy)

        elif key == ord('r') or key == ord('R'):
            image_copy = current_image.copy()
            draw_runways(image_copy, runways)
            cv2.imshow("Runway Designer", image_copy)

    cv2.destroyAllWindows()

    if runways:
        save = input("\nSave runways before exit? (y/n): ").strip().lower()
        if save == 'y':
            output_path = args.output if args.output else DEFAULT_OUTPUT_PATH
            save_runways_to_file(runways, output_path)

    print("\nExiting Runway Designer.")


if __name__ == "__main__":
    main()
