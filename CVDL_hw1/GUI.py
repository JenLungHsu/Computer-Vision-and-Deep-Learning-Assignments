import sys
import cv2
import numpy as np
import glob
import re
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QComboBox, QMessageBox, QLineEdit, QGridLayout, QGroupBox
from q2_onboard import show_result_onboard
from q2_vertical import show_result_vertical

class CvDlHwGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CvDl Homework GUI")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize image paths and variables for calibration parameters
        self.image_files = []
        self.objpoints = []
        self.imgpoints = []
        self.intrinsic_matrix = None
        self.dist_coeffs = None
        self.rvecs = []
        self.tvecs = []

        # Setup UI
        self.initUI()
    
    def initUI(self):
        # Calibration Group
        calibration_group = QGroupBox("1. Calibration")
        calibration_layout = QVBoxLayout()

        self.load_folder_button = QPushButton("Load Folder", self)
        self.load_folder_button.clicked.connect(self.load_folder)

        self.find_corners_button = QPushButton("1.1 Find Corners", self)
        self.find_corners_button.clicked.connect(self.find_corners)

        self.find_intrinsic_button = QPushButton("1.2 Find Intrinsic", self)
        self.find_intrinsic_button.clicked.connect(self.find_intrinsic)

        self.image_selector = QComboBox()
        self.image_selector.addItems([f"Image {i+1}" for i in range(15)])  #range(len(self.image_files))])

        self.find_extrinsic_button = QPushButton("1.3 Find Extrinsic", self)
        self.find_extrinsic_button.clicked.connect(self.find_extrinsic)

        self.find_distortion_button = QPushButton("1.4 Find Distortion", self)
        self.find_distortion_button.clicked.connect(self.find_distortion)

        self.show_undistortion_button = QPushButton("1.5 Show Undistortion", self)
        self.show_undistortion_button.clicked.connect(self.show_undistortion)

        calibration_layout.addWidget(self.load_folder_button)
        calibration_layout.addWidget(self.find_corners_button)
        calibration_layout.addWidget(self.find_intrinsic_button)
        calibration_layout.addWidget(self.image_selector)
        calibration_layout.addWidget(self.find_extrinsic_button)
        calibration_layout.addWidget(self.find_distortion_button)
        calibration_layout.addWidget(self.show_undistortion_button)
        calibration_group.setLayout(calibration_layout)

        # Augmented Reality
        ar_group = QGroupBox("2. Augmented Reality")
        ar_layout = QVBoxLayout()

        self.input_text = QLineEdit(self)

        self.show_words_board_button = QPushButton("2.1 Show Words on Board", self)
        self.show_words_board_button.clicked.connect(self.show_words_on_board)

        self.show_words_vertical_button = QPushButton("2.2 Show Words Vertical", self)
        self.show_words_vertical_button.clicked.connect(self.show_words_vertical)

        ar_layout.addWidget(self.input_text)
        ar_layout.addWidget(self.show_words_board_button)
        ar_layout.addWidget(self.show_words_vertical_button)
        ar_group.setLayout(ar_layout)

        # Stereo Disparity Map
        stereo_group = QGroupBox("3. Stereo Disparity Map")
        stereo_layout = QVBoxLayout()

        self.load_imageL_button = QPushButton("Load Image L", self)
        self.load_imageL_button.clicked.connect(self.load_imageL)

        self.load_imageR_button = QPushButton("Load Image R", self)
        self.load_imageR_button.clicked.connect(self.load_imageR)

        self.stereo_disparity_map_button = QPushButton("3.1 Stereo Disparity Map", self)
        self.stereo_disparity_map_button.clicked.connect(self.stereo_disparity_map)

        stereo_layout.addWidget(self.load_imageL_button)
        stereo_layout.addWidget(self.load_imageR_button)
        stereo_layout.addWidget(self.stereo_disparity_map_button)
        stereo_group.setLayout(stereo_layout)

        # SIFT
        sift_group = QGroupBox("4. SIFT")
        sift_layout = QVBoxLayout()

        self.load_image1_button = QPushButton("Load Image 1", self)
        self.load_image1_button.clicked.connect(self.load_image1)

        self.load_image2_button = QPushButton("Load Image 2", self)
        self.load_image2_button.clicked.connect(self.load_image2)

        self.show_keypoints_button = QPushButton("4.1 Keypoints", self)
        self.show_keypoints_button.clicked.connect(self.show_keypoints)

        self.show_matched_keypoints_button = QPushButton("4.2 Matched Keypoints", self)
        self.show_matched_keypoints_button.clicked.connect(self.show_matches)

        sift_layout.addWidget(self.load_image1_button)
        sift_layout.addWidget(self.load_image2_button)
        sift_layout.addWidget(self.show_keypoints_button)
        sift_layout.addWidget(self.show_matched_keypoints_button)
        sift_group.setLayout(sift_layout)

        # Main layout
        main_layout = QGridLayout()
        main_layout.addWidget(calibration_group, 0, 0)
        main_layout.addWidget(ar_group, 0, 1)
        main_layout.addWidget(stereo_group, 0, 2)
        main_layout.addWidget(sift_group, 1, 0, 1, 3)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.image_files = sorted(glob.glob(f"{folder_path}/*.bmp"), key=self.natural_sort_key)
            print("Loaded images from folder:", self.image_files)

    def find_corners(self):
        if not self.image_files:
            print("Please load images first.")
            return
        print("Finding corners...")
        
        chessboard_size = (11, 8)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        for fname in self.image_files:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001))
                self.imgpoints.append(corners2)
                cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
                cv2.imshow("Corners", img)
                cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_intrinsic(self):
        # 計算影像大小
        h, w = 2048, 2048

        # 計算內參矩陣、畸變係數、旋轉向量和位移向量
        ret, self.intrinsic_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, (w, h), None, None)

        if self.intrinsic_matrix is None:
            QMessageBox.warning(self, "Warning", "Intrinsic matrix not computed yet.")
            return

        # 使用 numpy.array2string 格式化輸出
        formatted_matrix = np.array2string(self.intrinsic_matrix, precision=6, separator=', ')
        display_text = f"Intrinsic matrix (camera matrix):\n{formatted_matrix}"

        # 顯示彈出窗口
        msg = QMessageBox()
        msg.setWindowTitle("Intrinsic Matrix")
        msg.setText(display_text)
        msg.exec_()

    def find_extrinsic(self):
        print("Calculating extrinsic matrices...")
        if not self.rvecs or not self.tvecs:
            print("Please calculate intrinsic parameters first.")
            return

        index = self.image_selector.currentIndex()
        rvec = self.rvecs[index]
        tvec = self.tvecs[index]
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        self.extrinsic_matrix = np.hstack((rotation_matrix, tvec))
        # QMessageBox.information(self, f"Extrinsic Matrix for Image {index+1}", str(extrinsic_matrix))

        if self.extrinsic_matrix is None:
            QMessageBox.warning(self, "Warning", "Extrinsic matrix not computed yet.")
            return

        # 使用 numpy.array2string 格式化輸出
        formatted_matrix = np.array2string(self.extrinsic_matrix, precision=6, separator=', ')
        display_text = f"Extrinsic matrix:\n{formatted_matrix}"

        # 顯示彈出窗口
        msg = QMessageBox()
        msg.setWindowTitle("Extrinsic Matrix")
        msg.setText(display_text)
        msg.exec_()

    def find_distortion(self):
        print("Distortion coefficients:")
        # QMessageBox.information(self, "Distortion Coefficients", str(self.dist_coeffs))

        if self.dist_coeffs is None:
            QMessageBox.warning(self, "Warning", "Distortion coefficients not computed yet.")
            return

        # 使用 numpy.array2string 格式化輸出
        formatted_matrix = np.array2string(self.dist_coeffs, precision=6, separator=', ')
        display_text = f"Distortion coefficients:\n{formatted_matrix}"

        # 顯示彈出窗口
        msg = QMessageBox()
        msg.setWindowTitle("Distortion coefficients")
        msg.setText(display_text)
        msg.exec_()
        
    def show_undistortion(self):
        print("Displaying undistorted images...")
        if self.intrinsic_matrix is None or self.dist_coeffs is None:
            QMessageBox.warning(self, "Warning", "Please calculate intrinsic parameters first.")
            return

        index = self.image_selector.currentIndex()
        fname = self.image_files[index]
        img = cv2.imread(fname)
        undistorted_img = cv2.undistort(img, self.intrinsic_matrix, self.dist_coeffs)

        # 添加標籤到影像
        img_with_text = img.copy()
        undistorted_with_text = undistorted_img.copy()
        cv2.putText(img_with_text, "Distorted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(undistorted_with_text, "Undistorted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 水平合併加了標籤的原始影像和無畸變影像
        combined_img = cv2.hconcat([img_with_text, undistorted_with_text])

        # 顯示合併影像
        cv2.imshow(f"Distorted and Undistorted Chessboard Image {index+1}", combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_words_on_board(self):
        print("Displaying words on board...")
        # Use code from `q2.py` to display "words on board" using `projectPoints`
        text = self.input_text.text()
        show_result_onboard(text)

    def show_words_vertical(self):
        print("Displaying words vertically...")
        # Use code from `q2.py` to display words vertically using `projectPoints`
        text = self.input_text.text()
        show_result_vertical(text)

    def load_imageL(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image L", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.imageL = cv2.imread(file_path)

    def load_imageR(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image R", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.imageR = cv2.imread(file_path)

    def stereo_disparity_map(self):
        print("Calculating and displaying stereo disparity map...")
        
        left_image = cv2.cvtColor(self.imageL, cv2.COLOR_BGR2GRAY)
        right_image = cv2.cvtColor(self.imageR, cv2.COLOR_BGR2GRAY)

        if left_image is None or right_image is None:
            raise ValueError("請確認左右影像的路徑是否正確。")

        # 計算視差圖
        stereo = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=11)
        disparity = stereo.compute(left_image, right_image)
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity_normalized = np.uint8(disparity_normalized)

        # 将图像缩小为 50%
        left_image_resized = cv2.resize(self.imageL, (left_image.shape[1] // 6, left_image.shape[0] // 6))
        right_image_resized = cv2.resize(self.imageR, (right_image.shape[1] // 6, right_image.shape[0] // 6))
        disparity_resized = cv2.resize(disparity_normalized, (disparity_normalized.shape[1] // 6, disparity_normalized.shape[0] // 6))
        
        # 顯示三個獨立視窗，並設定每個視窗的位置
        cv2.imshow("ImgL", left_image_resized)
        cv2.imshow("ImgR", right_image_resized)
        cv2.imshow("Disparity Map (Colored)", disparity_resized)
        
        # 設定視窗位置
        cv2.moveWindow("ImgL", 50, 50)  # 左影像位置
        cv2.moveWindow("ImgR", 400, 50)  # 右影像位置
        cv2.moveWindow("Disparity Map (Colored)", 750, 50)  # 彩色視差圖位置

        # 等待按鍵事件來關閉所有視窗
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_image1(self):
        # 選擇左影像
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image 1", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.image1 = cv2.imread(file_path)
            # 調整影像大小
            resized_image = cv2.resize(self.image1, (600, int(self.image1.shape[0] * (600 / self.image1.shape[1]))))
            # 在獨立視窗中顯示載入的影像 1
            cv2.imshow("Loaded Image 1", resized_image)
            cv2.waitKey(0)
            cv2.destroyWindow("Loaded Image 1")

    def load_image2(self):
        # 選擇右影像
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image 2", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.image2 = cv2.imread(file_path)
            # 調整影像大小
            resized_image = cv2.resize(self.image2, (600, int(self.image2.shape[0] * (600 / self.image2.shape[1]))))
            # 在獨立視窗中顯示載入的影像 2
            cv2.imshow("Loaded Image 2", resized_image)
            cv2.waitKey(0)
            cv2.destroyWindow("Loaded Image 2")

    def show_keypoints(self):
        # 初始化 SIFT 檢測器
        sift = cv2.SIFT_create()

        # 判斷並顯示 keypoints
        if self.image1 is not None and self.image2 is None:
            # 只有 image1
            gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            keypoints1, _ = sift.detectAndCompute(gray1, None)
            image1_with_keypoints = cv2.drawKeypoints(gray1, keypoints1, None, color=(0, 255, 0))
            resized_image1 = cv2.resize(image1_with_keypoints, (600, int(image1_with_keypoints.shape[0] * (600 / image1_with_keypoints.shape[1]))))
            cv2.imshow("Image 1 with Keypoints", resized_image1)
            cv2.waitKey(0)
            cv2.destroyWindow("Image 1 with Keypoints")

        elif self.image2 is not None and self.image1 is None:
            # 只有 image2
            gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
            keypoints2, _ = sift.detectAndCompute(gray2, None)
            image2_with_keypoints = cv2.drawKeypoints(gray2, keypoints2, None, color=(0, 255, 0))
            resized_image2 = cv2.resize(image2_with_keypoints, (600, int(image2_with_keypoints.shape[0] * (600 / image2_with_keypoints.shape[1]))))
            cv2.imshow("Image 2 with Keypoints", resized_image2)
            cv2.waitKey(0)
            cv2.destroyWindow("Image 2 with Keypoints")

        elif self.image1 is not None and self.image2 is not None:
            # 兩張影像都有
            gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
            keypoints1, _ = sift.detectAndCompute(gray1, None)
            keypoints2, _ = sift.detectAndCompute(gray2, None)

            # 繪製特徵點
            image1_with_keypoints = cv2.drawKeypoints(gray1, keypoints1, None, color=(0, 255, 0))
            image2_with_keypoints = cv2.drawKeypoints(gray2, keypoints2, None, color=(0, 255, 0))

            # 調整大小
            resized_image1 = cv2.resize(image1_with_keypoints, (600, int(image1_with_keypoints.shape[0] * (600 / image1_with_keypoints.shape[1]))))
            resized_image2 = cv2.resize(image2_with_keypoints, (600, int(image2_with_keypoints.shape[0] * (600 / image2_with_keypoints.shape[1]))))

            # 水平拼接
            combined_image = cv2.hconcat([resized_image1, resized_image2])

            # 顯示合併後的結果
            cv2.imshow("Combined Image with Keypoints", combined_image)
            cv2.waitKey(0)
            cv2.destroyWindow("Combined Image with Keypoints")

        else:
            print("請先載入至少一張影像。")

    def show_matches(self):
        # 確保兩張影像均已載入
        if self.image1 is not None and self.image2 is not None:
            gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
            
            # 初始化 SIFT 檢測器
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
            keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
            
            # 使用 BFMatcher 進行匹配
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            
            # 根據距離過濾出好的匹配點
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m])

            # 繪製匹配點
            matched_image = cv2.drawMatchesKnn(gray1, keypoints1, gray2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # 顯示匹配結果在獨立的 OpenCV 視窗
            cv2.imshow("Matched Keypoints", matched_image)
            
            # 等待按下任意鍵關閉窗口
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("請先載入左影像和右影像。")


    @staticmethod
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


# Run Application
app = QApplication(sys.argv)
window = CvDlHwGUI()
window.show()
sys.exit(app.exec_())
