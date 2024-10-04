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
    QSizePolicy,
    QMessageBox,
    QSplitter    
)

import logging
from PyQt5.QtGui import QPixmap, QImage, QIntValidator
from PyQt5.QtCore import Qt
from moviepy.editor import VideoFileClip, concatenate_videoclips
import subprocess 
from detect_filter import *
import pickle

logging.basicConfig(
    level=logging.DEBUG, format="%(levelname)s - Line: %(lineno)d - %(message)s"
)

class CenteredCheckBox(QWidget):
    def __init__(self, parent=None, set_checked=False):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(set_checked)
        self.checkbox.setFixedSize(120, 120)  # Set the size to 120x120 pixels
        layout.addWidget(self.checkbox, alignment=Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove any margins
        self.setLayout(layout)


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


def find_contiguous_sequences(frames, min_length):
    # Sort the frames to ensure they are in order
    frames.sort()

    # To store the contiguous sequences
    contiguous_sequences = []
    current_sequence = [frames[0]]

    # Iterate through the sorted frames and find contiguous sequences
    for i in range(1, len(frames)):
        # Check if the current frame is contiguous with the previous frame
        if frames[i] == frames[i - 1] + 1:
            current_sequence.append(frames[i])
        else:
            # If not contiguous, store the current sequence and start a new one
            if len(current_sequence) >= min_length:
                contiguous_sequences.append(current_sequence)
            current_sequence = [frames[i]]

    # Don't forget to add the last sequence
    if len(current_sequence) >= min_length:
        contiguous_sequences.append(current_sequence)

    return contiguous_sequences


class VideoSceneEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        master_layout = QVBoxLayout()
        splitter = QSplitter(Qt.Vertical)

        # Input video
        input_layout_widget = QWidget()
        input_layout = QHBoxLayout(input_layout_widget)
        input_layout.addWidget(QLabel("Input Video:"))
        self.input_video = QLineEdit()
        # add initial value
        self.input_video.setText("")
        input_layout.addWidget(self.input_video)
        self.input_video_btn = QPushButton("Browse")
        self.input_video_btn.clicked.connect(self.browse_input_video)
        input_layout.addWidget(self.input_video_btn)
        splitter.addWidget(input_layout_widget)

        # Output path
        output_layout_widget = QWidget()
        output_layout = QHBoxLayout(output_layout_widget)
        output_layout.addWidget(QLabel("Output Path:"))
        self.output_path = QLineEdit()
        self.output_path.setText(
            "D:\\ini\\UserSettings\\Google\\Config\\LandingZone\\screened-and-selected"
        )
        output_layout.addWidget(self.output_path)
        self.output_path_btn = QPushButton("Browse")
        self.output_path_btn.clicked.connect(self.browse_output_path)
        output_layout.addWidget(self.output_path_btn)
        splitter.addWidget(output_layout_widget)

        # CSV file
        csv_layout_widget = QWidget()
        csv_layout = QHBoxLayout(csv_layout_widget)
        csv_layout.addWidget(QLabel("CSV File:"))
        self.csv_file = QLineEdit()
        self.csv_file.setText("")
        csv_layout.addWidget(self.csv_file)
        self.csv_file_btn = QPushButton("Browse")
        self.csv_file_btn.clicked.connect(self.browse_csv_file)
        csv_layout.addWidget(self.csv_file_btn)
        splitter.addWidget(csv_layout_widget)

        # Table
        self.table = QTableWidget()
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Images", "play", "Keep?"])
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        splitter.addWidget(self.table)

        # Create QHBox layout for buttons
        button_layout_widget = QWidget()
        button_layout = QHBoxLayout(button_layout_widget)

        # Generate Scenes button
        self.generate_scenes_csv_btn = QPushButton("Generate Scenes File")
        self.generate_scenes_csv_btn.setMinimumSize(150, 50)
        self.generate_scenes_csv_btn.clicked.connect(self.generate_scenes_csv)
        button_layout.addWidget(self.generate_scenes_csv_btn)

        # parameter for min min of each scene
        self.min_scene_length_line_edit = QLineEdit()
        # set to restrict to numbers
        self.min_scene_length_line_edit.setValidator(QIntValidator())
        button_layout.addWidget(self.min_scene_length_line_edit)

        # Load Scenes From CSV button
        self.load_scenes_btn = QPushButton("Load Scenes From CSV")
        self.load_scenes_btn.setMinimumSize(150, 50)
        self.load_scenes_btn.clicked.connect(self.load_scenes)
        button_layout.addWidget(self.load_scenes_btn)

        # Keep selected scenes button
        self.keep_btn = QPushButton("Keep Selected Scenes")
        self.keep_btn.setMinimumSize(250, 50)  # Set the minimum size (width, height)
        self.keep_btn.clicked.connect(self.keep_selected_scenes)
        button_layout.addWidget(self.keep_btn)

        # Load No Face Ranges button
        self.load_no_face_btn = QPushButton("Load No Face Ranges")
        self.load_no_face_btn.setMinimumSize(
            250, 50
        )  # Set the minimum size (width, height)
        self.load_no_face_btn.clicked.connect(self.load_no_face_ranges)
        button_layout.addWidget(self.load_no_face_btn)

        splitter.addWidget(button_layout_widget)

        # Create another similar table
        self.table_no_face = QTableWidget()
        self.table_no_face.verticalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.table_no_face.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.table_no_face.setColumnCount(3)
        self.table_no_face.setHorizontalHeaderLabels(
            ["Images", "play", "Remove?"]
        )
        self.table_no_face.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        splitter.addWidget(self.table_no_face)

        master_layout.addWidget(splitter)
        self.setLayout(master_layout)
        self.setWindowTitle("Video Scene Editor")
        self.show()

    def overlaps_with_excluded_scenes(self, range, excluded_scenes):
        # check if range overlaps with any of the excluded scenes
        for excluded_range in excluded_scenes:
            # start point lies in excluded range
            if float(range[0]) >= float(excluded_range[0]) and float(range[0]) <= float(excluded_range[1]):
                print("scene {} overlaps with excluded scene: {} to {}".format(range, excluded_range[0], excluded_range[1]))
                return True
            # end point lies in excluded range
            elif float(range[1]) >= float(excluded_range[0]) and float(range[1]) <= float(excluded_range[1]):
                print("scene {} overlaps with excluded scene: {} to {}".format(range, excluded_range[0], excluded_range[1]))
                return True
            # the range itself engulfs the excluded range
            elif float(range[0]) <= float(excluded_range[0]) and float(range[1]) >= float(excluded_range[1]):
                print(
                    "scene {} overlaps with excluded scene: {} to {}".format(
                        range, excluded_range[0], excluded_range[1]
                    )
                )
                return True
            # total engulfed
            elif float(range[0]) >= float(excluded_range[0]) and float(range[1]) <= float(excluded_range[1]):
                return True
            else:
                print("Does not overlap with excluded scene: ", range, excluded_range)
                return False

    def load_no_face_ranges(self):
        input_video = self.input_video.text()
        video = VideoFileClip(self.input_video.text())
        fps = video.fps

        # load ranges from pickle file if exist. Filename should use basename of input video and exist in the same folder as the input video
        input_video_basename = os.path.basename(input_video)
        input_video_folder = os.path.dirname(input_video)
        pickle_file = os.path.join(
            input_video_folder,
            os.path.splitext(input_video_basename)[0] + ".pkl",
        )
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as file:
                ranges = pickle.load(file)
        else:
            frames_list, fps =  get_frames_w_no_face(input_video, providers)
            frames_list_filtered = find_contiguous_sequences(frames_list, fps)
            # flatten the list
            frames_list_filtered = [item for sublist in frames_list_filtered for item in sublist] 

            ranges = create_time_range_list_from_frame_info(frames_list_filtered, fps)
            with open(pickle_file, "wb") as file:
                pickle.dump(ranges, file)

        excluded_scenes = []
        for i, scene_data in enumerate(self.scenes_data):
            if not self.table.cellWidget(i, 2).checkbox.isChecked():
                excluded_scenes.append(
                    (scene_data["start_time"], scene_data["end_time"])
                )
        post_filtered_range = []

        for i, iter_range in enumerate(ranges):

            if self.overlaps_with_excluded_scenes(iter_range, excluded_scenes): 
                continue
            else:
                post_filtered_range.append(iter_range)

        ranges = post_filtered_range

        # set row count of table
        self.table_no_face.setRowCount(len(post_filtered_range))
        # assuming ranges is a list of tuple, containing start and end time of each range
        # populate the table with the ranges similar to self.table
        self.no_face_data = []
        for i, iter_range in enumerate(post_filtered_range):
            self.no_face_data.append({
                "scene_number": i,
                "start_time": iter_range[0],
                "end_time": iter_range[1]
            })
            start_time = iter_range[0]
            end_time = iter_range[1]

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

            # Get start frame given start time
            start_frame = int(fps * iter_range[0])
            end_frame = int(fps * iter_range[1])

            # Get images for the scene
            images = self.get_scene_images(start_frame, end_frame)
            image_widget = QWidget()
            image_layout = QHBoxLayout(image_widget)
            image_layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins
            for image in images:
                label = QLabel()
                pixmap = QPixmap.fromImage(image)
                pixmap = pixmap.scaledToHeight(
                    250, Qt.SmoothTransformation
                )  # Set minimum height to 150 pixels
                label.setPixmap(pixmap)
                image_layout.addWidget(label)

            self.table_no_face.setCellWidget(i, 0, image_widget)
            self.table_no_face.setRowHeight(
                i, 260
            )  # Set row height to accommodate images plus some padding

            # add play button that plays the video from the range of start_frame to end_frame
            play_btn = QPushButton("Play ({}s - {}s)".format(start_time, end_time))
            self.table_no_face.setCellWidget(i, 1, play_btn)
            play_btn.clicked.connect(
                lambda checked, start_time=start_time, end_time=end_time: self.play_video(
                    start_time, end_time
                )
            )

            # Add checkbox (initially unchecked)
            centered_checkbox = CenteredCheckBox(set_checked=False)
            self.table_no_face.setCellWidget(i, 2, centered_checkbox)

        pass
    def browse_input_video(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video", "", "*"
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

    def load_scenes(self):
        csv_file = self.csv_file.text()
        df = pd.read_csv(csv_file)
        self.table.setRowCount(len(df))
        self.scenes_data = []  # Clear previous data

        for i, row in df.iterrows():
            scene_number = row["Scene Number"]
            start_frame = row["Start Frame"]
            end_frame = row["End Frame"]
            start_time = float(row["Start Time (seconds)"])
            end_time = float(row["End Time (seconds)"])

            ori_start_time = start_time
            ori_end_time = end_time
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
                    "start_time": ori_start_time,
                    "end_time": ori_end_time,
                }
            )

            # Get images for the scene
            images = self.get_scene_images(start_frame, end_frame)
            image_widget = QWidget()
            image_layout = QHBoxLayout(image_widget)
            image_layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins
            for image in images:
                label = QLabel()
                pixmap = QPixmap.fromImage(image)
                pixmap = pixmap.scaledToHeight(
                    250, Qt.SmoothTransformation
                )  
                label.setPixmap(pixmap)
                image_layout.addWidget(label)

            self.table.setCellWidget(i, 0, image_widget)
            self.table.setRowHeight(
                i, 260
            )  # Set row height to accommodate images plus some padding

            # add play button that plays the video from the range of start_frame to end_frame
            play_btn = QPushButton("Play ({}s - {}s)".format(start_time, end_time))
            self.table.setCellWidget(i, 1, play_btn)
            play_btn.clicked.connect(
                lambda checked, start_time=start_time, end_time=end_time: self.play_video(
                    start_time, end_time
                )
            )

            # Add checkbox (initially checked)
            centered_checkbox = CenteredCheckBox(set_checked=True)
            self.table.setCellWidget(i, 2, centered_checkbox)
            for i in range(3):
                self.table.horizontalHeader().setSectionResizeMode(
                    i, QHeaderView.Interactive
                )

    # define play_video function
    def play_video(self, start_time, end_time):
        # play the video from start_frame to end_frame
        video_file = self.input_video.text()
        # convert seconds to hh:mm:ss format

        logging.info("mpv --start=" + start_time + " --end=" + end_time + " " + video_file)
        # run mpv command to play the video
        result = subprocess.run(
            [
                "mpv",
                "--start=" + start_time,
                "--end=" + end_time,
                video_file,
            ],
            text=True,
            stderr=subprocess.PIPE,
        )

        if result.returncode == 0:
            logging.info("Subprocess ran successfully")
        else:
            logging.info(f"Subprocess failed with return code {result.returncode}")
            logging.info(f"Error output: {result.stderr}")
            logging.info(f"Std output: {result.stdout}")

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
                # Step 2: Convert from BGR to RGB
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Step 3: Convert the NumPy array to QImage
                height, width, channel = image_rgb.shape
                bytes_per_line = 3 * width  # 3 bytes per pixel for RGB
                q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

                images.append(q_image)

        video.release()
        return images

    def find_maximals(self, time_ranges):
        if not time_ranges:
            return []

        # Sort the time ranges based on the start time
        sorted_ranges = sorted(time_ranges, key=lambda x: x[0])

        consolidated = [sorted_ranges[0]]

        for current_start, current_end in sorted_ranges[1:]:
            last_start, last_end = consolidated[-1]

            if current_start <= last_end:
                # Overlap or connection found, update the end time
                consolidated[-1] = (last_start, max(last_end, current_end))
            else:
                # No overlap, add as a new range
                consolidated.append((current_start, current_end))

        return consolidated

    def keep_selected_scenes(self):
        video = VideoFileClip(self.input_video.text())
        scenes_to_keep = []

        scenes_to_remove = []        
        # check table_no_face checkboxes so that we only include the range for which checkbox is checked
        for i, scene_data in enumerate(self.no_face_data):
            # If the checkbox is checked, include the scene in the scene_to_remove list
            if self.table_no_face.cellWidget(i, 2).checkbox.isChecked():
                scenes_to_remove.append((scene_data["start_time"], scene_data["end_time"]))

        keep_ranges = []

        for i, scene_data in enumerate(self.scenes_data):
            # If the checkbox is checked, keep the scene
            if self.table.cellWidget(i, 2).checkbox.isChecked():
                start_frame = scene_data["start_frame"]
                end_frame = scene_data["end_frame"]
                start_time = start_frame / video.fps
                end_time = end_frame / video.fps

                # Consolidate all the scenes to remove by finding the maximals of their intervals
                # also ensures that  they are connected with each other when finding the maximals
                scenes_to_remove = self.find_maximals(scenes_to_remove)

                logging.info(f"Scene {i}: start_time={start_time}, end_time={end_time}")

                scene_clip = video.subclip(start_time, end_time)
                keep_this_scene_due_to_no_overlap = True
                # keep a list of tuples to keep track of subclips start and end relative to the video
                for remove_start, remove_end in scenes_to_remove:
                    logging.info(f"Checking removal range: remove_start={remove_start}, remove_end={remove_end}")
                    # Case #1 - Remove range is completely within the scene, and boundaries don't match
                    if remove_start > start_time and remove_end < end_time:
                        logging.info(f"Remove range is completely within the scene: {remove_start} to {remove_end}")
                        # Split the scene into two parts
                        scenes_to_keep.append(
                            video.subclip(start_time, remove_start)
                        )
                        keep_ranges.append((start_time, remove_start))
                        logging.info(f"Keep range added: {start_time} to {remove_start}")
                        scenes_to_keep.append(
                            video.subclip(remove_end, end_time)
                        )
                        keep_ranges.append((remove_end, end_time))
                        logging.info(f"Keep range added: {remove_end} to {end_time}")
                        keep_this_scene_due_to_no_overlap = False
                    # case #2 - remove end stops in the middle of the scene, remove start is outside of scene
                    elif remove_start <= start_time and remove_end > start_time and remove_end < end_time:
                        logging.info(f"Remove end stops in the middle of the scene: {remove_end}")
                        scenes_to_keep.append(
                            video.subclip(remove_end, end_time)
                        )
                        keep_ranges.append((remove_end, end_time))
                        logging.info(f"Keep range added: {remove_end} to {end_time}")
                        keep_this_scene_due_to_no_overlap = False
                    # case #3 - remove start is in the middle of the scene, and remove end is outside of scene
                    elif remove_start > start_time and remove_start < end_time and remove_end >= end_time:
                        logging.info(f"Remove start is in the middle of the scene: {remove_start}")
                        scenes_to_keep.append(
                            video.subclip(start_time, remove_start)
                        )
                        keep_ranges.append((start_time, remove_start))
                        logging.info(f"Keep range added: {start_time} to {remove_start}")
                        keep_this_scene_due_to_no_overlap = False

                    # case #4 - remove range engulfs the entire scene
                    elif remove_start < start_time and remove_end > end_time:
                        keep_this_scene_due_to_no_overlap = False
                    # case #5 - no overlap with this scene to remove, but may overlap with other scene to remove in the scenes_to_remove list
                    else:
                        logging.info(f"No overlap with this scene to remove: {remove_start} to {remove_end}")
                        pass  # move on
                if keep_this_scene_due_to_no_overlap:
                    logging.info("Keeping this whole scene: start_time={start_time}, end_time={end_time}")
                    scenes_to_keep.append(scene_clip)
                    keep_ranges.append((start_time, end_time))

        # check length of scenes_to_keep against keep_ranges
        if len(scenes_to_keep) != len(keep_ranges):
            logging.warning("Warning: scenes_to_keep and keep_ranges length mismatch")
            logging.info(f"scenes_to_keep: {len(scenes_to_keep)}, keep_ranges: {len(keep_ranges)}")
        else:
            # print start and end time  for all scenes_to_keep
            for i, scene in enumerate(scenes_to_keep):
                start_time = keep_ranges[i][0]
                end_time = keep_ranges[i][1]
                logging.info(f"Scene {i}: start_time={start_time}, end_time={end_time}")

        if scenes_to_keep:
            final_clip = concatenate_videoclips(scenes_to_keep)
            input_file = self.input_video.text()
            base_filename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(self.output_path.text(), f"{base_filename}_edited.mp4")
            final_clip.write_videofile(output_file)
            logging.info(f"Video saved to {output_file}")
        else:
            logging.info("No scenes selected to keep. The output video would be empty.")

        video.close()
        self.close()

    def generate_scenes_csv(self):
        # check if scenes file already specified
        if self.csv_file.text() != "":
            # show warning dialog

            # show warning dialog
            reply = QMessageBox.question(
                self, 'Warning', 
                "Scenes file has been specified. Do you want to replace it?", 
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply == QMessageBox.No: return
            scenes_file = self.csv_file.text()

        else:
            # folder of the input video
            folder = os.path.dirname(self.input_video.text())
            # get basename without extension
            basename = os.path.splitext(os.path.basename(self.input_video.text()))[0]
            scenes_file = folder +  "\\" + basename + ".csv"

        input_video = self.input_video.text()

        if os.path.exists(scenes_file):
            os.remove(scenes_file)

        subprocess.run(
            [
                "D:\Installed\Anaconda\envs\scenedetection_python\Scripts\scenedetect.exe",
                "-i",
                input_video,
                # "--merge-last-scene",
                "list-scenes",
                "-f",
                scenes_file,
                "-s",
                "detect-adaptive",
            ]
        )
        self.csv_file.setText(scenes_file)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = VideoSceneEditor()
    sys.exit(app.exec_())
