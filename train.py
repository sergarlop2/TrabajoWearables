# -*- coding: utf-8 -*-
"""
Activity Classifier using 1D Convolutional Neural Network
Based on provided padel classifier
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# Configuration
window_size = 80
activities = ["baseball", "bolos", "boxeo", "golf", "tenis", "reposo"]
n_classes = len(activities)
normalize = True


# Load and preprocess all data
def load_all_data():
    all_data = []
    for activity in activities:
        df = pd.read_csv(f"datasets/datos_wii_{activity}.csv")
        df["actividad"] = activities.index(activity)
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


# Load the data
datos = load_all_data()


# Function to create windows of sensor data
def create_windows(data, window_size):
    windows = []
    labels = []
    for _, group in data.groupby("actividad"):
        for i in range(0, len(group) - window_size, window_size // 4):  # 25% overlap
            window = group.iloc[i : i + window_size]
            if len(window) == window_size:
                features = np.concatenate(
                    [
                        window["pitch"].values,
                        window["roll"].values,
                        window["yaw"].values,
                        window["x_accel"].values,
                        window["y_accel"].values,
                        window["z_accel"].values,
                    ]
                )
                windows.append(features)
                labels.append(window["actividad"].iloc[0])
    return np.array(windows), np.array(labels)


# Create windowed dataset
X, y = create_windows(datos, window_size)


# Ensure no overlap between train, val, and test sets
def split_data(X, y, test_size, val_size, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Test set split
    test_split = int(test_size * len(indices))
    val_split = int(val_size * (len(indices) - test_split))

    test_indices = indices[:test_split]
    val_indices = indices[test_split : test_split + val_split]
    train_indices = indices[test_split + val_split :]

    return (
        X[train_indices],
        y[train_indices],
        X[val_indices],
        y[val_indices],
        X[test_indices],
        y[test_indices],
    )


# Split the data
test_size = 0.2
val_size = 0.15
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size, val_size)


# Reshape data for CNN
def reshape_for_cnn(X):
    samples = X.shape[0]
    X = X.reshape(samples, 6, window_size)
    return X


X_train = reshape_for_cnn(X_train)
X_val = reshape_for_cnn(X_val)
X_test = reshape_for_cnn(X_test)

if normalize:
    norm_values = {"mean": [], "std": []}
    
    for i in range(6):
        mean = X_train[:, i, :].mean()
        norm_values["mean"].append(mean)
        std = X_train[:, i, :].std()
        norm_values["std"].append(std)
        
        X_train[:, i, :] = (X_train[:, i, :] - mean) / std
        X_val[:, i, :] = (X_val[:, i, :] - mean) / std
        X_test[:, i, :] = (X_test[:, i, :] - mean) / std
        
    # Save normalization values in a JSON file
    with open("normalization_values.json", "w") as f:
        json.dump(norm_values, f)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Create data loaders
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Define the CNN model
class ActivityNet(nn.Module):
    def __init__(self, n_classes, window_size):
        super().__init__()
        # Calculate the size of features after convolutions
        # First conv: output_size = input_size - kernel_size + 1
        # MaxPool: output_size = input_size // 2
        size_after_conv1 = window_size - 3 + 1  # kernel_size=3
        size_after_conv2 = size_after_conv1 - 3 + 1  # kernel_size=3
        size_after_pool = size_after_conv2 // 2
        self.flatten_size = 4 * size_after_pool  # 4 is the number of filters in conv2

        self.layer1 = nn.Sequential(
            nn.Conv1d(6, 8, kernel_size=3), nn.ReLU(), nn.BatchNorm1d(8)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Dropout(0.3),
            nn.MaxPool1d(2),
        )
        self.layer3 = nn.Sequential(nn.Linear(self.flatten_size, n_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out, 1)
        out = self.layer3(out)
        return out


# Initialize model with window_size parameter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ActivityNet(n_classes, window_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
best_val_acc = 0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_losses = []
    train_accs = []

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        accuracy = accuracy_score(batch_y.cpu().numpy(), predictions)

        train_losses.append(loss.item())
        train_accs.append(accuracy)

    # Validation phase
    model.eval()
    val_losses = []
    val_accs = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            accuracy = accuracy_score(batch_y.cpu().numpy(), predictions)

            val_losses.append(loss.item())
            val_accs.append(accuracy)

    # Calculate epoch metrics
    train_loss = np.mean(train_losses)
    train_acc = np.mean(train_accs)
    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)

    # Store history
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_activity_model.pt")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Test phase
model.eval()
test_predictions = []
test_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        test_predictions.extend(predictions)
        test_labels.extend(batch_y.cpu().numpy())

# Calculate and display test metrics
test_acc = accuracy_score(test_labels, test_predictions)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Display confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=activities,
    yticklabels=activities,
)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display classification report
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(test_labels, test_predictions, target_names=activities))
