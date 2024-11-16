import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score

# data dir
data_dir = '/root/merged'

# clear unnecessary folders
for root, dirs, files in os.walk(data_dir):
    if '.ipynb_checkpoints' in dirs:
        shutil.rmtree(os.path.join(root, '.ipynb_checkpoints'))

# data transformation
transform = transforms.Compose([
    transforms.Resize((462, 620)),
    transforms.ToTensor(),
])

# Using ImageFolder
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# split train,val, test
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Using DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


class ODConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding='same', groups=1, reduction=0.0625,
                 kernel_num=4):
        super(ODConv, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 if padding == 'same' else 0
        self.groups = groups
        self.attention_channel = max(int(in_planes * reduction), 16)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, self.attention_channel, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.attention_channel)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(self.attention_channel, in_planes, kernel_size=1, padding=0, bias=True)
        self.kernel_weights = nn.Parameter(
            torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.kernel_weights, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        attention = self.avgpool(x)
        attention = self.fc1(attention)
        attention = self.bn1(attention)
        attention = self.relu(attention)
        channel_attention = torch.sigmoid(self.fc2(attention))
        x = x * channel_attention
        aggregate_weight = torch.sum(self.kernel_weights, dim=0)
        output = nn.functional.conv2d(x, aggregate_weight, stride=self.stride, padding=self.padding, groups=self.groups)
        return output


class GNConv(nn.Module):
    def __init__(self, in_channels, dim, order=3, kernel_size=7):
        super(GNConv, self).__init__()
        self.order = order
        self.kernel_size = kernel_size
        self.conv_in = nn.Conv2d(in_channels, dim * 2, kernel_size=1, padding=0)
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.conv_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.scale = 1.0

    def forward(self, x):
        x_proj = self.conv_in(x)
        pwa, abc = torch.split(x_proj, x_proj.shape[1] // 2, dim=1)
        dw_abc = self.depthwise_conv(abc) * self.scale
        x = pwa * dw_abc
        x = self.conv_out(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# Model definition
class BuildModel(nn.Module):
    def __init__(self, input_shape=(3, 462, 620), num_classes=5):
        super(BuildModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.odconv1 = ODConv(in_planes=16, out_planes=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.gnconv = GNConv(in_channels=32, dim=64)
        self.bn3 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 231 * 310, 256)  # Based on input size
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.odconv1(x)
        x = self.bn2(x)
        x = self.maxpool1(x)
        x = self.gnconv(x)
        x = self.bn3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# Init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BuildModel().to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# model train
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate acc
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # Validation process
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_predictions = []
    val_true_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_predictions.extend(predicted.cpu().numpy())  # Prediction
            val_true_labels.extend(labels.cpu().numpy())  # real labels

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')  # Calculate F1 score

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')

# Test process
model.eval()  # Switch to evaluation mode
test_loss = 0.0
test_correct = 0
test_total = 0
test_predictions = []
test_true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        test_predictions.extend(predicted.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader)
test_acc = 100 * test_correct / test_total
test_f1 = f1_score(test_true_labels, test_predictions, average='weighted')

print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.4f}')

for true_labels, predicted_labels in test_predictions[:5]:
    print(f'Test True: {true_labels.numpy()}, Predicted: {predicted_labels.numpy()}')