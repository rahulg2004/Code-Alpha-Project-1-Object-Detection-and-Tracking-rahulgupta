import cv2
import time
import argparse
import os
import urllib.request
import numpy as np
from collections import Counter
from ultralytics import YOLO
from sort import Sort


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "yolo11n.pt")

MODEL_URL = (
    "https://github.com/ultralytics/assets/"
    "releases/download/v8.3.0/yolo11n.pt"
)


def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print("YOLO model found in models folder.")
        return

    print("YOLO model not found.")
    print("Downloading yolo11n.pt into models folder...")
    print()

    try:
        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print()
        print("YOLO model downloaded successfully.")
        print(f"Model location: {MODEL_PATH}")

    except Exception as error:
        print()
        print("Error downloading YOLO model.")
        print(error)

        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        raise


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CodeAlpha Object Detection and Tracking"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="0 for webcam or path to video file"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.40,
        help="Minimum detection confidence"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output/tracked_output.mp4",
        help="Output video path"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save processed video"
    )

    return parser.parse_args()


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    box1_area = max(
        0,
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    box2_area = max(
        0,
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union_area = (
        box1_area +
        box2_area -
        intersection_area
    )

    if union_area <= 0:
        return 0

    return intersection_area / union_area


def create_video_writer(
    output_path,
    width,
    height,
    fps
):
    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    return cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )


def main():
    args = parse_arguments()

    print("=" * 60)
    print("CODEALPHA")
    print("OBJECT DETECTION AND TRACKING")
    print("=" * 60)
    print()

    download_model()

    print()
    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    print("YOLO model loaded successfully.")
    print()

    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    video_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if video_fps <= 0:
        video_fps = 30.0

    tracker = Sort(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3
    )

    writer = None

    if args.save:
        writer = create_video_writer(
            args.output,
            width,
            height,
            video_fps
        )

    recording = args.save

    previous_time = time.time()

    total_frames = 0

    total_unique_ids = set()

    print("Controls:")
    print("Q - Quit")
    print("S - Save screenshot")
    print("R - Toggle recording")
    print()

    while True:

        success, frame = cap.read()

        if not success:
            break

        total_frames += 1

        current_time = time.time()

        results = model(
            frame,
            conf=args.confidence,
            verbose=False
        )

        result = results[0]

        detections = []

        detection_classes = []

        detection_confidences = []

        if result.boxes is not None:

            for box in result.boxes:

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                confidence = float(
                    box.conf[0]
                    .cpu()
                    .numpy()
                )

                class_id = int(
                    box.cls[0]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = coordinates

                detections.append(
                    [
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence
                    ]
                )

                detection_classes.append(
                    class_id
                )

                detection_confidences.append(
                    confidence
                )

        if len(detections) > 0:

            detections_array = np.array(
                detections,
                dtype=float
            )

        else:

            detections_array = np.empty(
                (0, 5),
                dtype=float
            )

        tracks = tracker.update(
            detections_array
        )

        tracked_objects = []

        for track in tracks:

            x1, y1, x2, y2, track_id = track

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            track_id = int(track_id)

            total_unique_ids.add(
                track_id
            )

            best_iou = 0

            best_class_id = None

            best_confidence = 0

            track_box = [
                x1,
                y1,
                x2,
                y2
            ]

            for index, detection in enumerate(
                detections
            ):

                detection_box = [
                    detection[0],
                    detection[1],
                    detection[2],
                    detection[3]
                ]

                iou = calculate_iou(
                    track_box,
                    detection_box
                )

                if iou > best_iou:

                    best_iou = iou

                    best_class_id = (
                        detection_classes[index]
                    )

                    best_confidence = (
                        detection_confidences[index]
                    )

            if best_class_id is not None:

                class_name = model.names[
                    best_class_id
                ]

            else:

                class_name = "Object"

            tracked_objects.append(
                {
                    "id": track_id,
                    "class_name": class_name,
                    "confidence": best_confidence,
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                }
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = (
                f"{class_name} | "
                f"ID: {track_id} | "
                f"{best_confidence:.0%}"
            )

            label_width = max(
                180,
                len(label) * 10
            )

            label_top = max(
                0,
                y1 - 30
            )

            cv2.rectangle(
                frame,
                (
                    x1,
                    label_top
                ),
                (
                    x1 + label_width,
                    y1
                ),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (
                    x1 + 5,
                    y1 - 8
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        elapsed_time = (
            current_time -
            previous_time
        )

        if elapsed_time > 0:

            fps = (
                1 /
                elapsed_time
            )

        else:

            fps = 0

        previous_time = current_time

        object_names = []

        for detection_class in detection_classes:

            object_names.append(
                model.names[
                    detection_class
                ]
            )

        object_counts = Counter(
            object_names
        )

        panel_height = 160

        cv2.rectangle(
            frame,
            (10, 10),
            (310, panel_height),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Objects: {len(tracks)}",
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Unique IDs: {len(total_unique_ids)}",
            (25, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        if recording:

            recording_text = "REC"

        else:

            recording_text = "LIVE"

        cv2.putText(
            frame,
            recording_text,
            (25, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        y_offset = 190

        for object_name, count in (
            object_counts.items()
        ):

            if y_offset >= height - 10:
                break

            text = (
                f"{object_name}: "
                f"{count}"
            )

            cv2.putText(
                frame,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            y_offset += 25

        if recording:

            if writer is None:

                writer = create_video_writer(
                    args.output,
                    width,
                    height,
                    video_fps
                )

            writer.write(frame)

        cv2.imshow(
            "CodeAlpha - Object Detection and Tracking",
            frame
        )

        key = (
            cv2.waitKey(1) &
            0xFF
        )

        if key == ord("q"):

            break

        elif key == ord("s"):

            os.makedirs(
                "output",
                exist_ok=True
            )

            screenshot_path = (
                f"output/"
                f"screenshot_"
                f"{total_frames}.jpg"
            )

            cv2.imwrite(
                screenshot_path,
                frame
            )

            print(
                f"Screenshot saved: "
                f"{screenshot_path}"
            )

        elif key == ord("r"):

            recording = not recording

            if recording:

                if writer is None:

                    writer = create_video_writer(
                        args.output,
                        width,
                        height,
                        video_fps
                    )

                print(
                    "Recording started."
                )

            else:

                print(
                    "Recording stopped."
                )

    cap.release()

    if writer is not None:

        writer.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)
    print(
        f"Frames processed: "
        f"{total_frames}"
    )

    print(
        f"Unique tracked IDs: "
        f"{len(total_unique_ids)}"
    )

    if os.path.exists(
        args.output
    ):

        print(
            f"Output video: "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()