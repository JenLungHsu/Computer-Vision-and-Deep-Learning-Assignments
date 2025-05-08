import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
from train_dcgan import FlatImageDataset, Generator, Discriminator

class DCGANApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DcGAN_mnist")
        self.setGeometry(100, 100, 300, 200)

        # 主界面
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 布局
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # 按鈕區域
        self.load_mnist_button = QPushButton("1. Show Training Images")
        self.load_mnist_button.clicked.connect(self.load_mnist)
        self.layout.addWidget(self.load_mnist_button)

        self.load_model_button = QPushButton("2. Show Model Structure")
        self.load_model_button.clicked.connect(self.show_model_structure)
        self.layout.addWidget(self.load_model_button)

        self.load_loss_button = QPushButton("3. Show Training Loss")
        self.load_loss_button.clicked.connect(self.show_loss_plot)
        self.layout.addWidget(self.load_loss_button)

        self.show_images_button = QPushButton("4. Inference")
        self.show_images_button.clicked.connect(self.show_real_and_fake_images)
        self.layout.addWidget(self.show_images_button)

        # 初始化屬性
        self.generator = None
        self.discriminator = None
        self.latent_dim = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_mnist(self):
        """加載並顯示 MNIST 訓練圖片及增強圖片"""
        self.original_transform = transforms.Compose([
            transforms.Grayscale(),  # 確保是灰階
            transforms.Resize((64, 64)),  # 將圖像調整為 64x64
            transforms.ToTensor()
        ])
        self.augmented_transform = transforms.Compose([
            transforms.Grayscale(),  # 確保是灰階
            transforms.Resize((64, 64)),  # 將圖像調整為 64x64
            transforms.RandomRotation(60),  # 增強數據
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 將像素標準化到 [-1, 1]
        ])

        # 資料路徑
        data_path = "/Users/xurenlong/Desktop/CVDL_hw2/Q2_images/data/mnist"
        dataset_original = FlatImageDataset(root=data_path, transform=self.original_transform)

        dataloader_original = torch.utils.data.DataLoader(dataset_original, batch_size=64, shuffle=True)

        # 隨機挑選 64 張原始圖片
        original_images, _ = next(iter(dataloader_original))

        # 將張量轉換為 PIL 圖片進行數據增強
        original_images_pil = [transforms.ToPILImage()(img) for img in original_images]
        augmented_images = torch.stack([self.augmented_transform(img) for img in original_images_pil])

        # 顯示原始圖片
        original_grid = make_grid(original_images, nrow=8, normalize=True, scale_each=True)
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.title("Training Dataset (Original)")
        plt.axis("off")
        plt.imshow(original_grid.permute(1, 2, 0).numpy())

        # 顯示增強圖片
        augmented_grid = make_grid(augmented_images, nrow=8, normalize=True, scale_each=True)
        plt.subplot(1, 2, 2)
        plt.title("Training Dataset (Augmented)")
        plt.axis("off")
        plt.imshow(augmented_grid.permute(1, 2, 0).numpy())

        plt.show()

    def show_model_structure(self):
        """顯示生成器和辨別器模型結構"""
        try:
            # 加載生成器
            self.generator = Generator().to(self.device)
            generator_path = "/Users/xurenlong/Desktop/CVDL_hw2/dcgan_model_generator.pth"  # 固定路徑
            self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
            self.generator.eval()

            # 加載辨別器
            self.discriminator = Discriminator().to(self.device)
            discriminator_path = "/Users/xurenlong/Desktop/CVDL_hw2/dcgan_model_discriminator.pth"  # 固定路徑
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device))
            self.discriminator.eval()

            # 打印模型結構
            print("Generator Structure:")
            print(self.generator)
            print("\nDiscriminator Structure:")
            print(self.discriminator)

        except Exception as e:
            print(f"Failed to Load Model Structures: {e}")

    def show_loss_plot(self):
        """顯示固定路徑的訓練損失圖"""
        loss_path = "/Users/xurenlong/Desktop/CVDL_hw2/dcgan_model_loss_curve.png"  # 固定路徑
        try:
            img = plt.imread(loss_path)
            plt.figure()
            plt.title("Training Loss")
            plt.axis("off")
            plt.imshow(img)
            plt.show()
        except Exception as e:
            print(f"Failed to Load Loss Plot: {e}")

    def show_real_and_fake_images(self):
        """顯示真實圖像和生成器生成的假圖像"""
        if self.generator is None:
            print("Please Load a Generator Model First")
            return

        # 加載 MNIST 圖片
        data_path = "/Users/xurenlong/Desktop/CVDL_hw2/Q2_images/data/mnist"
        dataset = FlatImageDataset(root=data_path, transform=self.original_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        real_images, _ = next(iter(dataloader))

        # 生成假圖像
        with torch.no_grad():
            noise = torch.randn(64, self.latent_dim, 1, 1, device=self.device)
            fake_images = self.generator(noise).cpu()

        # 顯示真實和生成圖片在同一個畫布
        plt.figure(figsize=(16, 8))

        # 顯示真實圖片
        real_grid = make_grid(real_images, nrow=8, normalize=True, scale_each=True)
        plt.subplot(1, 2, 1)
        plt.title("Real Images")
        plt.axis("off")
        plt.imshow(real_grid.permute(1, 2, 0).numpy())

        # 顯示生成圖片
        fake_grid = make_grid(fake_images, nrow=8, normalize=True, scale_each=True)
        plt.subplot(1, 2, 2)
        plt.title("Fake Images")
        plt.axis("off")
        plt.imshow(fake_grid.permute(1, 2, 0).numpy())

        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DCGANApp()
    window.show()
    sys.exit(app.exec_())
