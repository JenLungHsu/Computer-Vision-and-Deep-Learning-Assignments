import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget

class SIFT_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIFT Feature Matching")
        self.setGeometry(100, 100, 300, 200)
        
        # 初始化影像
        self.image1 = None
        self.image2 = None
        
        # 建立UI
        self.initUI()
    
    def initUI(self):
        # 按鈕
        self.loadImage1Btn = QPushButton("Load Image 1", self)
        self.loadImage1Btn.clicked.connect(self.load_image1)

        self.loadImage2Btn = QPushButton("Load Image 2", self)
        self.loadImage2Btn.clicked.connect(self.load_image2)

        self.showKeypointsBtn = QPushButton("4.1 Show Keypoints", self)
        self.showKeypointsBtn.clicked.connect(self.show_keypoints)

        self.showMatchesBtn = QPushButton("4.2 Show Matches", self)
        self.showMatchesBtn.clicked.connect(self.show_matches)

        # 垂直佈局
        layout = QVBoxLayout()
        layout.addWidget(self.loadImage1Btn)
        layout.addWidget(self.loadImage2Btn)
        layout.addWidget(self.showKeypointsBtn)
        layout.addWidget(self.showMatchesBtn)

        # 設置中心窗口
        centralWidget = QWidget(self)
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

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

# 啟動應用程式
app = QApplication(sys.argv)
window = SIFT_GUI()
window.show()
sys.exit(app.exec_())
