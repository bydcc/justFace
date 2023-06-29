import os

import cv2
import numpy as np
from sio import sio
from insightface.app import FaceAnalysis
from pathlib import Path


class VideoFrameExtractor:
    def __init__(self, frame_frequency=12):
        self.frame_frequency = frame_frequency

    def extract_frames(self, video_folder, output_folder):
        video_folder = Path(video_folder)
        output_folder = Path(output_folder)
        # 确保输出目录存在
        Path.mkdir(output_folder, exist_ok=True)
        for file_path in Path.iterdir(video_folder):
            print(file_path, file_path.suffix)
            if file_path.suffix in [".mp4", ".avi"]:
                print(1111)
                capture = cv2.VideoCapture(str(file_path))
                count = 0
                frame_count = 0
                while True:
                    success, image = capture.read()
                    if not success:
                        break
                    if count % self.frame_frequency == 0:
                        cv2.imwrite(str(output_folder / "{}_frame{}.jpg".format(
                            file_path.name, frame_count)), image)
                        frame_count += 1
                    count += 1
                capture.release()


class FaceExtractor:
    def __init__(self):
        self.app = FaceAnalysis(
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    @staticmethod
    def get_image_names_paths(folder_path):
        for file_path in Path.iterdir(folder_path):
            if file_path.suffix in [".jpg", ".jpeg", ".png"]:
                yield file_path

    def extract_and_save_images(self, image_folder, output_folder):
        image_folder = Path(image_folder)
        output_folder = Path(output_folder)
        Path.mkdir(output_folder, exist_ok=True)
        for file_path in self.get_image_names_paths(image_folder):
            image = cv2.imread(str(file_path))
            faces = self.app.get(image)
            
            for index, face in enumerate(faces):
                coord = face['bbox'].astype(int)
                x1, y1, x2, y2 = coord
                cropped_image = image[y1:y2, x1:x2]
                if cropped_image.size !=  0:
                    output_path = os.path.join(
                        output_folder, f"{file_path.name}_{index}.png")
                    cv2.imwrite(output_path, cropped_image)
