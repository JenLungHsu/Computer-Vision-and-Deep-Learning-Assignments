import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import VGG19_BN_Weights
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

# 訓練參數
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./saved_weights"
os.makedirs(SAVE_DIR, exist_ok=True)

# 資料增強與加載資料集
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 建立模型
model = models.vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1).to(DEVICE)

# 修改分類層，將輸出類別數改為 10
model.classifier[-1] = nn.Linear(in_features=4096, out_features=10).to(DEVICE)

# 定義損失函數與優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 訓練與驗證函數
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    best_model_weights = None

    for epoch in range(epochs):
        # 訓練階段
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)

        train_losses.append(total_train_loss / len(train_loader))
        train_accuracies.append(correct_train / total_train)

        # 驗證階段
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_val += predicted.eq(labels).sum().item()
                total_val += labels.size(0)

        val_losses.append(total_val_loss / len(val_loader))
        val_accuracies.append(correct_val / total_val)

        # 更新 Learning Rate
        scheduler.step()

        # 儲存最佳模型
        if val_accuracies[-1] > best_val_accuracy:
            best_val_accuracy = val_accuracies[-1]
            best_model_weights = model.state_dict()

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")
        print(f"Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")
        print(f"Current Learning Rate: {scheduler.get_last_lr()}")

    # 儲存最佳權重
    torch.save(best_model_weights, os.path.join(SAVE_DIR, "vgg19_bn_best.pth"))
    print("Best model saved with accuracy:", best_val_accuracy)

    return train_losses, val_losses, train_accuracies, val_accuracies

# 執行訓練與驗證
train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(
    model, train_loader, val_loader, criterion, optimizer, EPOCHS
)

# 繪製準確率與損失圖表
plt.figure(figsize=(12, 5))

# 損失圖
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 準確率圖
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# 儲存圖表
plt.savefig(os.path.join(SAVE_DIR, "training_results.png"))
plt.show()
