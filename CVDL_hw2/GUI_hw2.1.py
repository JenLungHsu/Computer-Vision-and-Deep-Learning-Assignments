import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
)
from PyQt5.QtGui import QPixmap
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os
import torch
from PIL import Image

class CIFAR10App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CIFAR10 VGG19 GUI")
        self.setGeometry(100, 100, 300, 200)

        # Main layout
        self.layout = QVBoxLayout()

        # Buttons
        self.btn_show_augment = QPushButton("1. Show Augmented Images")
        self.btn_show_model = QPushButton("2. Show Model Structure")
        self.btn_show_training = QPushButton("3. Load and Show Training Results")
        self.btn_inference = QPushButton("4. Load Model and Inference on Image")

        self.layout.addWidget(self.btn_show_augment)
        self.layout.addWidget(self.btn_show_model)
        self.layout.addWidget(self.btn_show_training)
        self.layout.addWidget(self.btn_inference)

        # Connect buttons
        self.btn_show_augment.clicked.connect(self.show_augmented_images)
        self.btn_show_model.clicked.connect(self.show_model_structure)
        self.btn_show_training.clicked.connect(self.load_and_show_training_results)
        self.btn_inference.clicked.connect(self.load_model_and_infer_image)

        # Image label
        self.image_label = QLabel()
        self.image_label.setFixedSize(200, 150)
        self.layout.addWidget(self.image_label)

        # Prediction label
        self.prediction_label = QLabel("Prediction: None")
        self.layout.addWidget(self.prediction_label)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        # Model and transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg19_bn(num_classes=10).to(self.device)
        self.transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    def show_augmented_images(self):
        # 選擇資料夾
        dataset_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", os.path.expanduser("~"))
        if not dataset_path:
            print("No folder selected.")
            return

        print(f"Selected dataset folder: {dataset_path}")

        # 獲取資料夾中的所有圖片檔案
        image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) < 9:
            print("Dataset must contain at least 9 images.")
            return

        # 加載前9張圖片
        images = [Image.open(img_path).convert("RGB") for img_path in image_files[:9]]

        # 從文件名提取類別名稱
        labels = [os.path.splitext(os.path.basename(f))[0].split('.')[0] for f in image_files[:9]]

        # 定義數據增強
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30)
        ])

        # 應用增強（在 PIL 圖片上操作）
        augmented_images = [transform(img) for img in images]

        # 在新視窗中顯示圖片
        fig, axes = plt.subplots(3, 3, figsize=(6, 6))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(augmented_images[i])
            ax.axis('off')
            ax.set_title(f"{labels[i]}")
        plt.tight_layout()
        plt.show(block=True)

    def show_model_structure(self):
        from torchsummary import summary
        summary(self.model, (3, 32, 32))

    def load_and_show_training_results(self):
        # 固定圖片路徑
        file_path = "/Users/xurenlong/Desktop/CVDL_hw2/training_results.png"  # 替換為你的圖片路徑
        if os.path.exists(file_path):
            # pixmap = QPixmap(file_path)
            # self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), aspectRatioMode=1))

            # 在新視窗中顯示圖片
            img = Image.open(file_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title("Training Result Image")
            plt.show(block=True)
        else:
            print(f"Image not found at {file_path}")

    def load_model_and_infer_image(self):
        # 固定模型權重檔案路徑
        weight_path = "/Users/xurenlong/Desktop/CVDL_hw2/vgg19_bn_best.pth"
        if os.path.exists(weight_path):
            try:
                # 載入模型權重
                self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
                self.model.eval()
                print("Model weights loaded successfully.")
            except FileNotFoundError:
                print(f"File not found at {weight_path}")
            except RuntimeError as e:
                print(f"Error loading model weights: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
        else:
            print(f"Model weights file not found at {weight_path}")

        # 選擇圖片進行推論
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image for Inference", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            try:
                # 圖片預處理
                image = Image.open(file_path).convert("RGB")
                image = self.transforms(image).unsqueeze(0).to(self.device)

                # 模型推論
                with torch.no_grad():
                    outputs = self.model(image)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    pred = torch.argmax(probs, dim=1).item()

                # 顯示結果於 GUI
                class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
                self.prediction_label.setText(f"Prediction: {class_names[pred]}")

                # 顯示概率分佈圖
                plt.figure(figsize=(10, 5))
                plt.bar(class_names, probs.cpu().numpy().squeeze(), color='skyblue')
                plt.xticks(rotation=45)
                plt.xlabel("Class")
                plt.ylabel("Probability")
                plt.title("Probability of Each Class")
                for i, p in enumerate(probs.cpu().numpy().squeeze()):
                    plt.text(i, p + 0.01, f"{p:.2f}", ha="center")
                plt.tight_layout()
                plt.show(block=True)

                # Show image and results
                pixmap = QPixmap(file_path).scaled(128, 128)
                self.image_label.setPixmap(pixmap)
                print(f"Predicted Class: {class_names[pred]}")

            except Exception as e:
                print(f"Error during inference: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CIFAR10App()
    window.show()
    sys.exit(app.exec_())
