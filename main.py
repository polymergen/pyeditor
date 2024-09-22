import sys
import os
import cv2
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QCheckBox,
    QSizePolicy
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from moviepy.editor import VideoFileClip, concatenate_videoclips
import subprocess 
from detect_filter import *


class ImageWidget(QWidget):
    def __init__(self, images, parent=None):
        super().__init__(parent)
        self.images = images
        self.image_labels = []
        self.initUI()

    def initUI(self):
        self.image_layout = QHBoxLayout()
        for image in self.images:
            label = QLabel()
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            pixmap = QPixmap(image)
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            self.image_layout.addWidget(label)
            self.image_labels.append(label)
        self.setLayout(self.image_layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjustImageSizes()

    def adjustImageSizes(self):
        for label in self.image_labels:
            pixmap = label.pixmap()
            if pixmap:
                label.setPixmap(
                    pixmap.scaled(
                        label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )


class VideoSceneEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Input video
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Video:"))
        self.input_video = QLineEdit()
        # add initial value
        self.input_video.setText("D:\\awan\\iCloudDrive\\CloudData\\Settings\\Config\\Google\\UserSettings\\Mapdata\\OrgBorderland\\Egral\\StrongestGroundp1.mif")
        input_layout.addWidget(self.input_video)
        self.input_video_btn = QPushButton("Browse")
        self.input_video_btn.clicked.connect(self.browse_input_video)
        input_layout.addWidget(self.input_video_btn)
        layout.addLayout(input_layout)

        # Output path
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Path:"))
        self.output_path = QLineEdit()
        self.output_path.setText(
            "D:\\ini\\UserSettings\\Google\\Config\\LandingZone\\screened-and-selected"
        )
        output_layout.addWidget(self.output_path)
        self.output_path_btn = QPushButton("Browse")
        self.output_path_btn.clicked.connect(self.browse_output_path)
        output_layout.addWidget(self.output_path_btn)
        layout.addLayout(output_layout)

        # CSV file
        csv_layout = QHBoxLayout()
        csv_layout.addWidget(QLabel("CSV File:"))
        self.csv_file = QLineEdit()
        csv_layout.addWidget(self.csv_file)
        self.csv_file_btn = QPushButton("Browse")
        self.csv_file_btn.clicked.connect(self.browse_csv_file)
        csv_layout.addWidget(self.csv_file_btn)
        layout.addLayout(csv_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Scene Number", "Images", "play", "Keep?"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.table, 1)

        # Load No Face Ranges button
        self.load_no_face_btn = QPushButton("Load No Face Ranges")
        self.load_no_face_btn.setMinimumSize(250, 150)  # Set the minimum size (width, height)
        self.load_no_face_btn.clicked.connect(self.load_no_face_ranges)
        layout.addWidget(self.load_no_face_btn)

        # Keep selected scenes button
        self.keep_btn = QPushButton("Keep Selected Scenes")
        self.keep_btn.setMinimumSize(250, 150)  # Set the minimum size (width, height)
        self.keep_btn.clicked.connect(self.keep_selected_scenes)
        layout.addWidget(self.keep_btn)

        # Create another similar table
        self.table_no_face = QTableWidget()
        self.table_no_face.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Scene Number", "Images", "play", "Keep?"]
        )
        self.table_no_face.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_no_face.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.table_no_face)

        self.setLayout(layout)
        self.setWindowTitle("Video Scene Editor")
        self.show()

    def load_no_face_ranges(self):
        input_video = self.input_video.text()
        ranges =  get_frames_w_no_face(input_video, providers)

        # assuming ranges is a list of tuple, containing start and end time of each range
        # populate the table with the ranges similar to self.table
        for i, range in enumerate(ranges):
            start_time = range[0]
            end_time = range[1]
            
            start_time = (
                str(int(start_time // 3600)).zfill(2)
                + ":"
                + str(int((start_time % 3600) // 60)).zfill(2)
                + ":"
                + str(int(start_time % 60)).zfill(2)
            )
            end_time = (
                str(int(end_time // 3600)).zfill(2)
                + ":"
                + str(int((end_time % 3600) // 60)).zfill(2)
                + ":"
                + str(int(end_time % 60)).zfill(2)
            )
            
            self.table_no_face.setItem(i, 0, QTableWidgetItem(str(i)))


            # Get start frame given start time
            video = cv2.VideoCapture(input_video)
            fps = video.get(cv2.CAP_PROP_FPS)
            start_frame = int(fps * range[0])
            end_frame = int(fps * range[1])
            
            # Get images for the scene
            images = self.get_scene_images(start_frame, end_frame)
            image_widget = QWidget()
            image_layout = QHBoxLayout(image_widget)
            image_layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins
            for image in images:
                label = QLabel()
                pixmap = QPixmap(image)
                pixmap = pixmap.scaledToHeight(
                    250, Qt.SmoothTransformation
                )  # Set minimum height to 150 pixels
                label.setPixmap(pixmap)
                image_layout.addWidget(label)

            self.table_no_face.setCellWidget(i, 1, image_widget)
            self.table_no_face.setRowHeight(
                i, 260
            )  # Set row height to accommodate images plus some padding

            # add play button that plays the video from the range of start_frame to end_frame
            play_btn = QPushButton("Play ({}s - {}s)".format(start_time, end_time))
            self.table_no_face.setCellWidget(i, 2, play_btn)
            play_btn.clicked.connect(
                lambda checked, start_time=start_time, end_time=end_time: self.play_video(
                    start_time, end_time
                )
            )

            # Add checkbox (initially checked)
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.setFixedSize(120, 120)  # Set the size to 20x20 pixels
            self.table_no_face.setCellWidget(i, 3, checkbox)

    def browse_input_video(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video", "", "Video Files (*.mp4 *.avi)"
        )
        if filename:
            self.input_video.setText(filename)

    def browse_output_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_path.setText(path)

    def browse_csv_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )
        if filename:
            self.csv_file.setText(filename)
            self.load_scenes(filename)

    def load_scenes(self, csv_file):
        df = pd.read_csv(csv_file)
        self.table.setRowCount(len(df))
        self.scenes_data = []  # Clear previous data

        for i, row in df.iterrows():
            scene_number = row["Scene Number"]
            start_frame = row["Start Frame"]
            end_frame = row["End Frame"]
            start_time = float(row["Start Time (seconds)"])
            end_time = float(row["End Time (seconds)"])

            start_time = (
                str(int(start_time // 3600)).zfill(2)
                + ":"
                + str(int((start_time % 3600) // 60)).zfill(2)
                + ":"
                + str(int(start_time % 60)).zfill(2)
            )
            end_time = (
                str(int(end_time // 3600)).zfill(2)
                + ":"
                + str(int((end_time % 3600) // 60)).zfill(2)
                + ":"
                + str(int(end_time % 60)).zfill(2)
            )

            self.scenes_data.append(
                {
                    "scene_number": scene_number,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )

            self.table.setItem(i, 0, QTableWidgetItem(str(scene_number)))

            # Get images for the scene
            images = self.get_scene_images(start_frame, end_frame)
            image_widget = QWidget()
            image_layout = QHBoxLayout(image_widget)
            image_layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins
            for image in images:
                label = QLabel()
                pixmap = QPixmap(image)
                pixmap = pixmap.scaledToHeight(
                    250, Qt.SmoothTransformation
                )  # Set minimum height to 150 pixels
                label.setPixmap(pixmap)
                image_layout.addWidget(label)

            self.table.setCellWidget(i, 1, image_widget)
            self.table.setRowHeight(
                i, 260
            )  # Set row height to accommodate images plus some padding

            # add play button that plays the video from the range of start_frame to end_frame
            play_btn = QPushButton("Play ({}s - {}s)".format(start_time, end_time))
            self.table.setCellWidget(i, 2, play_btn)
            play_btn.clicked.connect(
                lambda checked, start_time=start_time, end_time=end_time: self.play_video(
                    start_time, end_time
                )
            )

            # Add checkbox (initially checked)
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.setFixedSize(120, 120)  # Set the size to 20x20 pixels
            self.table.setCellWidget(i, 3, checkbox)

    # define play_video function
    def play_video(self, start_time, end_time):
        # play the video from start_frame to end_frame
        video_file = self.input_video.text()
        # convert seconds to hh:mm:ss format

        print("mpv --start=" + start_time + " --end=" + end_time + " " + video_file)
        # run mpv command to play the video
        subprocess.run(
            [
                "mpv",
                "--start=" + start_time,
                "--end=" + end_time,
                video_file,
            ]
        )

    def get_scene_images(self, start_frame, end_frame):
        video = cv2.VideoCapture(self.input_video.text())
        scene_length = end_frame - start_frame + 1
        interval = scene_length // 4  # Divide the scene into 4 equal parts
        frames = [start_frame, start_frame + interval, start_frame + 2 * interval]
        images = []

        for frame in frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = video.read()
            if ret:
                filename = f"temp_frame_{frame}.jpg"
                cv2.imwrite(filename, img)
                images.append(filename)

        video.release()
        return images

    def keep_selected_scenes(self):
        video = VideoFileClip(self.input_video.text())
        scenes_to_keep = []

        for i, scene_data in enumerate(self.scenes_data):
            # If the checkbox is checked, keep the scene
            if self.table.cellWidget(i, 2).isChecked():
                start_frame = scene_data["start_frame"]
                end_frame = scene_data["end_frame"]
                start_time = start_frame / video.fps
                end_time = end_frame / video.fps
                scenes_to_keep.append(video.subclip(start_time, end_time))

        if scenes_to_keep:
            final_clip = concatenate_videoclips(scenes_to_keep)
            input_file = self.input_video.text()  # Assuming input_video is a QLineEdit containing the input file path
            base_filename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(self.output_path.text(), f"{base_filename}.mp4")
            final_clip.write_videofile(output_file)
            print(f"Video saved to {output_file}")
        else:
            print("No scenes selected to keep. The output video would be empty.")

        video.close()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = VideoSceneEditor()
    sys.exit(app.exec_())
