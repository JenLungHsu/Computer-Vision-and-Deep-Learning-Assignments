import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        自定義 Dataset，適用於沒有中間分類資料夾的資料集。
        
        參數：
        - root: 圖片所在的根目錄
        - transform: 圖片的變換方法
        """
        self.root = root
        self.transform = transform
        # 獲取目錄下的所有圖片路徑
        self.images = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")  # 確保圖片是 RGB 格式
        if self.transform:
            img = self.transform(img)
        return img, 0  # 返回圖像和虛擬標籤（不需要標籤時可以設為 0）

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: latent vector Z of size 100
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),  # Output: (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # Output: (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # Output: (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # Output: (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),     # Output: (3, 64, 64)
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: image of size (3, 64, 64)
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),  # Output: (64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # Output: (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # Output: (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # Output: (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),    # Output: (1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def train_dcgan(dataloader, generator, discriminator, num_epochs=5, device='cuda', lr=0.0002, beta1=0.5, latent_dim=100, model_save_path='dcgan_model'):
    """
    訓練 DCGAN 模型的函式。

    參數：
        dataloader: DataLoader, 包含訓練資料的 DataLoader。
        generator: nn.Module, 生成器模型。
        discriminator: nn.Module, 判別器模型。
        num_epochs: int, 訓練的總 epoch 數。
        device: str, 使用的裝置 (例如 'cuda' 或 'cpu')。
        lr: float, 學習率。
        beta1: float, Adam 優化器的 beta1 參數。
        latent_dim: int, 潛在空間的維度大小。
        model_save_path: str, 模型和損失圖保存的路徑。

    返回：
        G_losses: list, 每個訓練步驟的生成器損失。
        D_losses: list, 每個訓練步驟的判別器損失。
        generator: nn.Module, 訓練後的生成器模型。
        discriminator: nn.Module, 訓練後的判別器模型。
    """
    # 初始化損失函數與優化器
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    # 學習率調整器
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=5, gamma=0.7)
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=5, gamma=0.7)

    # 固定的隨機噪聲，用於監控生成器的生成效果
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    # 記錄損失
    G_losses = []
    D_losses = []

    # 保存最低生成器損失
    best_g_loss = float('inf')

    # 將模型移動到指定裝置
    generator.to(device)
    discriminator.to(device)

    print("Starting Training Loop...")

    # 訓練過程
    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(dataloader), 0):
            # 1. 更新 Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            discriminator.zero_grad()

            # 真實資料
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            real_labels = torch.full((b_size,), 0.9, dtype=torch.float, device=device)
            output = discriminator(real_images).view(-1)
            lossD_real = criterion(output, real_labels)
            lossD_real.backward()

            # 假資料
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            fake_labels = torch.full((b_size,), 0.1, dtype=torch.float, device=device)
            output = discriminator(fake_images.detach()).view(-1)
            lossD_fake = criterion(output, fake_labels)
            lossD_fake.backward()

            # 更新 Discriminator
            optimizerD.step()

            # 2. 更新 Generator: maximize log(D(G(z)))
            generator.zero_grad()
            output = discriminator(fake_images).view(-1)
            lossG = criterion(output, real_labels)  # 使用真實標籤，讓生成器騙過判別器
            lossG.backward()
            optimizerG.step()

            # 記錄損失
            G_losses.append(lossG.item())
            D_losses.append(lossD_real.item() + lossD_fake.item())

            # 保存最佳生成器模型
            if lossG.item() < best_g_loss:
                best_g_loss = lossG.item()
                torch.save(generator.state_dict(), f"{model_save_path}_generator.pth")
                torch.save(discriminator.state_dict(), f"{model_save_path}_discriminator.pth")

        # 調整學習率
        schedulerD.step()
        schedulerG.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss_D: {D_losses[-1]:.4f} | Loss_G: {G_losses[-1]:.4f}")

    # 繪製並保存損失曲線
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G Loss")
    plt.plot(D_losses, label="D Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{model_save_path}_loss_curve.png")
    plt.show()

    return G_losses, D_losses, generator, discriminator

if __name__ == "__main__":
    # 資料路徑
    # data_path = "/Users/xurenlong/Desktop/CVDL_hw2/Q2_images/data/mnist"

    # 資料轉換
    transform = transforms.Compose([
        transforms.Grayscale(),  # 確保是灰階
        transforms.Resize((64, 64)),  # 將圖像調整為 64x64
        transforms.RandomRotation(60),  # 增強數據
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 將像素標準化到 [-1, 1]
    ])

    # 載入資料集
    # dataset = FlatImageDataset(root=data_path, transform=transform)
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 示例調用函式：
    generator = Generator()
    discriminator = Discriminator()
    G_losses, D_losses, trained_G, trained_D = train_dcgan(dataloader, 
                                                        generator, 
                                                        discriminator,
                                                        num_epochs=20, 
                                                        device='cuda', 
                                                        lr=0.0002, 
                                                        beta1=0.5, 
                                                        latent_dim=100, 
                                                        model_save_path='dcgan_model')
