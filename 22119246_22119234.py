import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Ensure CuBLAS deterministic behavior

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import pathlib
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler
import uuid
import matplotlib.pyplot as plt
from collections import Counter

# Disable deterministic algorithms to avoid CuBLAS error
torch.use_deterministic_algorithms(False)

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Seed everything để đảm bảo kết quả tái hiện được
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# Data preparation
training_dir = pathlib.Path('/kaggle/input/radar-signal-classification/training_set')

# Giảm bớt data augmentation để phù hợp hơn
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_dataset = datasets.ImageFolder(root=str(training_dir), transform=val_transform)

# Stratified split để đảm bảo phân phối đồng đều
targets = np.array(full_dataset.targets)
class_counts = Counter(targets)
print(f"Class counts: {class_counts}")

# Tạo các indices theo từng class
class_indices = {cls: np.where(targets == cls)[0] for cls in range(len(full_dataset.classes))}

# Stratified split
train_indices = []
val_indices = []

for cls, indices in class_indices.items():
    np.random.shuffle(indices)
    split = int(0.85 * len(indices))
    train_indices.extend(indices[:split])
    val_indices.extend(indices[split:])

np.random.shuffle(train_indices)
np.random.shuffle(val_indices)

# Tạo dataset với các indices đã chia
train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

# Áp dụng transform sau khi đã chia dataset
train_dataset.dataset.transform = train_transform

num_classes = len(full_dataset.classes)

# Sử dụng batch size nhỏ hơn để giảm sự chênh lệch giữa các batch
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# ECA Module
class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs(np.log2(channels) / gamma + b))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# SE Module
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

# ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Mô hình với số tham số 297,723
class ImprovedRadarCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(ImprovedRadarCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(40, 40, kernel_size=3, padding=1, groups=40, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 72, kernel_size=1, bias=False),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            ECAModule(72),
            nn.MaxPool2d(2, 2),
            ResBlock(72, 72),
            nn.Conv2d(72, 72, kernel_size=3, padding=1, groups=72, bias=False),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            nn.Conv2d(72, 72, kernel_size=1, bias=False),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            SEBlock(72),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            ResBlock(72, 72),
            nn.Conv2d(72, 72, kernel_size=3, padding=1, groups=72, bias=False),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            nn.Conv2d(72, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Dropout(0.2),  # Tăng dropout từ 0.15 lên 0.2
            nn.Linear(128 * 4 * 4, 36),
            nn.BatchNorm1d(36),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(36, num_classes)
        )
        self.skip_connection = nn.Sequential(
            nn.Conv2d(40, 128, kernel_size=1, stride=8, bias=False),
            nn.BatchNorm2d(128),
        )
        self.skip_connection2 = nn.Sequential(
            nn.Conv2d(72, 128, kernel_size=1, stride=4, bias=False),
            nn.BatchNorm2d(128),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        x_skip1 = self.features[0:4](x)
        x = self.features[4:12](x_skip1)
        x_skip2 = self.features[12:14](x)
        x = self.features[14:](x_skip2)
        skip1 = self.skip_connection(x_skip1)
        skip2 = self.skip_connection2(x_skip2)
        x = x + skip1 + skip2
        x = F.relu(x)
        x = self.classifier(x)
        return x

# Hàm training
def train_with_augmentation(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, patience=15):  # Tăng epochs lên 100, giảm patience về 15
    best_acc = 0.0
    best_loss = float('inf')
    early_stop_counter = 0
    scaler = GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    max_grad_norm = 0.5

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    def mixup_data(x, y, alpha=0.1):  # Tăng alpha từ 0.05 lên 0.1
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def cutmix_data(x, y, alpha=0.1):  # Tăng alpha từ 0.05 lên 0.1
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam

    def rand_bbox(size, lam):
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            if epoch > 5:
                if np.random.rand() > 0.8:  # Trả lại ngưỡng 0.8
                    images, targets_a, targets_b, lam = mixup_data(images, labels)
                elif np.random.rand() > 0.8:  # Trả lại ngưỡng 0.8
                    images, targets_a, targets_b, lam = cutmix_data(images, labels)
                else:
                    targets_a, targets_b, lam = labels, labels, 1.0
            else:
                targets_a, targets_b, lam = labels, labels, 1.0

            optimizer.zero_grad()
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model based on Val Accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with accuracy: {best_acc:.4f}")

        # Early stopping based on Val Loss
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model_early_stop.pth')
            print(f"New best Val Loss: {best_loss:.4f}, model saved for early stopping.")
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. No improvement in Val Loss for {patience} epochs.")
                break

    # Plotting
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_plots_optimized.png')

    return model

# Setup and train
model = ImprovedRadarCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Tăng label smoothing từ 0.03 lên 0.05
optimizer = optim.AdamW(model.parameters(), lr=0.0004, weight_decay=1e-4)  # Giảm lr về 0.0004, tăng weight decay lên 1e-4
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=1, eta_min=1e-6
)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Tổng số tham số của mạng: {total_params}")

# Train with early stopping
model = train_with_augmentation(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, patience=25)

# Save final model
example_input = torch.randn(1, 3, 128, 128).to(device)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("22119246_22119234.pt")
print("Model saved as 22119246_22119234.pt")