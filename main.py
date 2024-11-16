import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score
from sklearn.utils import class_weight
from sklearn import preprocessing
import numpy as np
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
from models import ODConv, GNConv, SEBlock, ResidualSEBlock,BuildModel

# Data loading and preprocessing
try:
    xall = joblib.load('dataset/xall.p')
    yall = joblib.load('dataset/yall.p')
    print("Files loaded successfully with joblib")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Label encoding
le = preprocessing.LabelEncoder()
yall = le.fit_transform(yall)
num_classes = len(np.unique(yall))
print(np.unique(yall, return_counts=True), num_classes)

# Shuffle data
xall, yall = shuffle(xall, yall)

# Split dataset into train, validation, and test sets
x_train, x_temp, y_train, y_temp = train_test_split(xall, yall, test_size=0.17, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Print split results
print(f'Train size: {len(x_train)}, Validation size: {len(x_val)}, Test size: {len(x_test)}')

# Compute class weights
class_weight0 = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weight0, dtype=torch.float32)

# Convert to tensors and adjust dimensions
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32).permute(0, 3, 1, 2)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create data loaders
batch_size = 32
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = BuildModel(input_shape=(2, 80, 80), num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.00008)

# Training parameters
num_epochs = 300
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Start training
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc.item())

    # Validate
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels.data)

    val_loss = val_running_loss / len(val_dataset)
    val_acc = val_running_corrects.double() / len(val_dataset)
    print(f'Validation Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())

    # Early stopping
    if val_acc >= 0.9952:
        print(f'Validation accuracy reached {val_acc:.4f}, stopping training.')
        break

end_time = time.time()
total_time = end_time - start_time
print(f'Total training time: {total_time:.2f} seconds')

# Test the model
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_sum = np.trace(cm_normalized)
average_auc = cm_sum / cm.shape[0]
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print results
print('Test Accuracy:', accuracy)
print('Average AUC:', average_auc)
print('Recall:', recall)
print('F1 Score:', f1)

# Plot training and validation loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save model
model_dir = r'../model/'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'od_gnconv_res_se_model_final.pth')
torch.save(model.state_dict(), model_path)
print('Model saved to:', model_path)

# Save training history
history_df = pd.DataFrame({
    'train_loss': train_losses,
    'train_accuracy': train_accuracies,
    'val_loss': val_losses,
    'val_accuracy': val_accuracies
})
history_path = os.path.join(model_dir, 'history_subjecttype.p')
history_df.to_pickle(history_path)
print('History saved to:', history_path)
print(history_df.head(100))