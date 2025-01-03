import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sense_hat import SenseHat
import time
import json
import matplotlib.pyplot as plt

# Configuration
duracion = 5.0
num_datos = 165
window_size = 80
activities = ["baseball", "bolos", "boxeo", "golf", "tenis", "reposo"]
n_classes = len(activities)
normalize = True
norm_values_file = "normalization_values.json"

sense = SenseHat()

# Lista para guardar los datos de la grabacion
columnas = ["actividad", "t", "pitch", "roll", "yaw", "x_accel", "y_accel", "z_accel"]
nuevos_datos = []


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


# Function to preprocess new data (without labels)
def preprocess_data(data, window_size, normalize=True):
    # Convert the raw sensor data into windows (no labels involved)
    X = create_windows(data, window_size)
    X = reshape_for_cnn(X)

    if normalize:
        try:
            with open(norm_values_file, "r") as f:
                norm_values = json.load(f)
                means = np.array(norm_values["mean"])
                stds = np.array(norm_values["std"])
            for i in range(6):
                X[:, i, :] = (X[:, i, :] - means[i]) / stds[i]
        except FileNotFoundError:
            print("Error: Normalization values file not found.")

    X = torch.FloatTensor(X).to(device)
    return X


# Create windows from the raw sensor data (no labels)
def create_windows(data, window_size):
    windows = []
    for i in range(0, len(data) - window_size, window_size // 4):  # 25% overlap
        window = data[i : i + window_size]
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
    return np.array(windows)


# Reshape data for CNN
def reshape_for_cnn(X):
    samples = X.shape[0]
    X = X.reshape(samples, 6, window_size)
    return X


# Function to predict activity from new data
def predict_activity(model, data, window_size):
    X = preprocess_data(data, window_size)
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()


# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ActivityNet(n_classes, window_size).to(device)
model.load_state_dict(torch.load("best_activity_model.pt", weights_only=True, map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Ask user to press enter before starting recording
input("Pulsa enter cuando estes listo para hacer la demo: ")

print("Preparándose para grabar datos...")
for i in range(3, 0, -1):
    print(f"{i}...")
    time.sleep(1)

print("¡GRABANDO!")

# Registrar el tiempo de inicio
inicio = time.time()

while time.time() - inicio <= duracion * 1.1:
    # Obtener datos del sensor
    ori = sense.get_orientation()
    ace = sense.get_accelerometer_raw()

    # Calcular el tiempo actual desde el inicio
    t = time.time() - inicio

    # Cada fila se compone de:
    # [1 COL (actividad) +
    #  1 COL (t) +
    #  3 COLS (orientación) +
    #  3 COLS (aceleración)]
    datos_fila = [
        None,
        t,
        ori["pitch"],
        ori["roll"],
        ori["yaw"],
        ace["x"],
        ace["y"],
        ace["z"],
    ]

    # Agregar datos a la lista
    nuevos_datos.append(datos_fila)

# Convertimos los datos a dataframe
nuevos_datos = pd.DataFrame(nuevos_datos, columns=columnas)

# Validar el número de muestras
num_capturado = len(nuevos_datos)

if num_capturado >= num_datos:
    # Predict activities
    predictions = predict_activity(model, nuevos_datos, window_size)

    # Map predictions to activity labels
    predicted_activities = [activities[pred] for pred in predictions]

    # Display predicted activities
    print("Actividades:", predicted_activities)
    print("Demo finalizada.")
else:
    print(
        f"Error: No se han tomado suficientes datos ({num_capturado} muestras, mínimo requerido: {num_datos})."
    )
