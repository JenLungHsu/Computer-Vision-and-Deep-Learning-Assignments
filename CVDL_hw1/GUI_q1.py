import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt
import glob

class CvDlHwGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CvDlHw")
        self.setGeometry(100, 100, 800, 600)

        # Initialize images and other variables
        self.image1 = None
        self.image2 = None
        self.intrinsic_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None

        # Load image files
        self.images = sorted(glob.glob('/path/to/images/*.bmp'), key=self.natural_sort_key)
        self.setup_ui()

    def setup_ui(self):
        # Layout for buttons and labels
        layout = QVBoxLayout()

        # Load images
        load_folder_button = QPushButton("Load Folder", self)
        load_folder_button.clicked.connect(self.load_folder)

        self.load_image1_btn = QPushButton("Load Image_L")
        self.load_image1_btn.clicked.connect(self.load_image1)
        layout.addWidget(self.load_image1_btn)

        self.load_image2_btn = QPushButton("Load Image_R")
        self.load_image2_btn.clicked.connect(self.load_image2)
        layout.addWidget(self.load_image2_btn)

        # Calibration
        calibration_layout = QVBoxLayout()
        calibration_layout.addWidget(QLabel("1. Calibration"))

        find_corners_btn = QPushButton("1.1 Find corners")
        find_corners_btn.clicked.connect(self.find_corners)
        calibration_layout.addWidget(find_corners_btn)

        find_intrinsic_btn = QPushButton("1.2 Find intrinsic")
        find_intrinsic_btn.clicked.connect(self.find_intrinsic)
        calibration_layout.addWidget(find_intrinsic_btn)

        self.image_selector = QComboBox()
        self.image_selector.addItems([f"Image {i+1}" for i in range(len(self.images))])
        calibration_layout.addWidget(self.image_selector)

        find_extrinsic_btn = QPushButton("1.3 Find extrinsic")
        find_extrinsic_btn.clicked.connect(self.find_extrinsic)
        calibration_layout.addWidget(find_extrinsic_btn)

        find_distortion_btn = QPushButton("1.4 Find distortion")
        find_distortion_btn.clicked.connect(self.find_distortion)
        calibration_layout.addWidget(find_distortion_btn)

        undistort_btn = QPushButton("1.5 Show undistorted")
        undistort_btn.clicked.connect(self.show_undistorted)
        calibration_layout.addWidget(undistort_btn)

        # Add other areas as shown in the provided UI image

        # Add to main window layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.addLayout(calibration_layout)
        main_layout.addLayout(layout)
        self.setCentralWidget(main_widget)

    def load_image1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image 1", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.image1 = cv2.imread(file_path)

    def load_image2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image 2", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.image2 = cv2.imread(file_path)

    def find_corners(self):
        for fname in self.images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret:
                cv2.drawChessboardCorners(img, (11, 8), corners, ret)
                cv2.imshow("Corners", img)
                if cv2.waitKey(0) == ord('n'):  # Press 'n' for next image
                    continue
            else:
                QMessageBox.warning(self, "Warning", f"Corners not found in {fname}")

    def find_intrinsic(self):
        # Calibration code here to set intrinsic_matrix and dist_coeffs
        QMessageBox.information(self, "Intrinsic Matrix", str(self.intrinsic_matrix))

    def find_extrinsic(self):
        index = self.image_selector.currentIndex()
        rvec = self.rvecs[index]
        tvec = self.tvecs[index]
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        extrinsic_matrix = np.hstack((rotation_matrix, tvec))
        QMessageBox.information(self, f"Extrinsic Matrix for Image {index+1}", str(extrinsic_matrix))

    def find_distortion(self):
        QMessageBox.information(self, "Distortion Coefficients", str(self.dist_coeffs))

    def show_undistorted(self):
        for fname in self.images:
            img = cv2.imread(fname)
            undistorted_img = cv2.undistort(img, self.intrinsic_matrix, self.dist_coeffs)
            cv2.imshow("Undistorted", undistorted_img)
            cv2.waitKey(0)

    @staticmethod
    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CvDlHwGUI()
    window.show()
    sys.exit(app.exec_())
