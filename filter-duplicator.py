import sys
import cv2
import numpy as np
import insightface
import torch
from insightface.app import FaceAnalysis
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QScrollArea, QCheckBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import subprocess
import tempfile
from tqdm import tqdm
import os

TEMP = "D:\\ini\\UserSettings\\Google\\Config\\LandingZone\\temp"
class VideoProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.face_detector = FaceAnalysis(providers=['CUDAExecutionProvider'])
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))
        self.segments = []
        self.selected_segments = set()
        self.frames_with_faces = set()  # Store indices of frames with faces
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Video Processor')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Select video button
        self.select_btn = QPushButton('Select Video', self)
        self.select_btn.clicked.connect(self.select_video)
        layout.addWidget(self.select_btn)

        # Scroll area for segments
        scroll = QScrollArea()
        self.segment_widget = QWidget()
        self.segment_layout = QVBoxLayout(self.segment_widget)
        scroll.setWidget(self.segment_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Process button
        self.process_btn = QPushButton('Process Video', self)
        self.process_btn.clicked.connect(self.process_final_video)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)

    def select_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Video')
        if fname:
            self.video_path = fname
            self.analyze_video()

    def analyze_video(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize arrays to track face detection results
        face_detected = []
        frames = []

        # Process each frame
        for i in tqdm(range(total_frames), desc="Detecting faces"):
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.face_detector.get(frame)
            has_face = len(faces) > 0
            face_detected.append(has_face)
            frames.append(frame)
            if has_face:  # Store indices of frames with faces
                self.frames_with_faces.add(i)

        # Find segments without faces
        segments = []
        start_idx = None

        for i in range(len(face_detected)):
            if not face_detected[i] and start_idx is None:
                start_idx = i
            elif face_detected[i] and start_idx is not None:
                duration = i - start_idx
                if duration >= 5:  # Faceless-acceptable segment
                    segments.append({
                        'start': start_idx,
                        'end': i,
                        'frame': frames[start_idx],
                        'type': 'segment',
                        'start_in_seconds': start_idx / fps,
                        'end_in_seconds': i / fps,
                        'duration_in_seconds': i / fps - start_idx / fps
                    })
                else:  # Loner frames
                    # Duplicate from nearest face frames
                    prev_face_frame = frames[max(0, start_idx - 1)]
                    next_face_frame = frames[min(i, len(frames) - 1)]
                    mid_point = start_idx + duration // 2

                    for j in range(start_idx, mid_point):
                        frames[j] = prev_face_frame
                    for j in range(mid_point, i):
                        frames[j] = next_face_frame

                start_idx = None

        self.segments = segments
        self.frames = frames
        self.fps = fps

        # Display segments
        self.display_segments()
        self.process_btn.setEnabled(True)

    def display_segments(self):
        # Clear existing segments
        for i in reversed(range(self.segment_layout.count())): 
            self.segment_layout.itemAt(i).widget().setVisible(False)
            self.segment_layout.itemAt(i).widget().deleteLater()

        # Add new segments
        for i, segment in enumerate(self.segments):
            container = QWidget()
            layout = QVBoxLayout(container)

            # Convert frame to QPixmap
            frame = segment['frame']
            height, width = frame.shape[:2]
            bytes_per_line = 3 * width
            image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(400, 300, Qt.KeepAspectRatio)

            # Create and add widgets
            img_label = QLabel()
            img_label.setPixmap(pixmap)
            layout.addWidget(img_label)

            checkbox = QCheckBox(f"Include Segment {i+1} (Frames {segment['start']} to {segment['end']})")
            checkbox.stateChanged.connect(lambda state, idx=i: self.toggle_segment(state, idx))
            layout.addWidget(checkbox)

            self.segment_layout.addWidget(container)

    def toggle_segment(self, state, idx):
        if state == Qt.Checked:
            self.selected_segments.add(idx)
        else:
            self.selected_segments.discard(idx)

    def process_final_video(self):
        output_path = self.video_path.rsplit('.', 1)[0] + '_processed.mp4'

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save frames as images
            for i, frame in enumerate(tqdm(self.frames, desc="Saving frames")):
                cv2.imwrite(f"{TEMP}/frame_{i:06d}.jpg", frame)

            # Create list of frames to include
            self.included_frames = set(self.frames_with_faces)

            # Add frames from selected segments
            for i, segment in enumerate(self.segments):
                if i in self.selected_segments:
                    for frame_idx in range(segment['start'], segment['end']):
                        self.included_frames.add(frame_idx)

            # Create frame list file
            with open(f"{TEMP}/frames.txt", 'w') as f:
                for i in range(len(self.frames)):
                    if i in self.included_frames:
                        f.write(f"file 'frame_{i:06d}.jpg'\n")
                        f.write(f"duration {1/self.fps}\n")

            # Use ffmpeg to create final video
            subprocess.run([
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', f"{TEMP}/frames.txt",
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'medium',
                output_path
            ])

            # Process audio separately and merge
            self.process_audio(output_path)

        self.close()

    def process_audio(self, video_path):
        # If all segments are selected (or no segments exist), use full audio
        all_segments_kept = len(self.segments) == len(self.selected_segments)
        if all_segments_kept or not self.segments:
            # Just copy the entire audio track
            final_output = video_path.rsplit('.', 1)[0] + '_with_audio.mp4'
            subprocess.run([
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', self.video_path,  # original video for audio
                '-c:v', 'copy',
                '-c:a', 'aac',
                final_output
            ])
            os.remove(video_path)
            return

        # Create a concat file for the audio segments
        concat_file = f"{TEMP}\\concat.txt"
        last_start_time = 0
        with open(concat_file, 'w') as f:
            for i, segment in enumerate(self.segments):
                if i not in self.selected_segments:
                    segment_file = f"{TEMP}\\segment_{i}.aac"
                    duration = segment['start_in_seconds'] - last_start_time
                    subprocess.run([
                        'ffmpeg', '-y',
                        '-i', self.video_path,
                        '-ss', str(last_start_time),
                        '-t', str(duration),
                        '-vn', '-acodec', 'copy',
                        segment_file
                    ])
                    last_start_time = segment['end_in_seconds']
                    f.write(f"file '{segment_file}'\n")

        # Concatenate the audio segments
        final_audio = f"{TEMP}/final_audio.aac"
        subprocess.run([
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c:a', 'aac',
            final_audio
        ])


        # Merge final audio with video
        final_output = video_path.rsplit('.', 1)[0] + '_with_audio.mp4'
        subprocess.run([
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', final_audio,
            '-c:v', 'copy',
            '-c:a', 'aac',
            final_output
        ])

        # Clean up
        os.remove(video_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoProcessor()
    ex.show()
    sys.exit(app.exec_())