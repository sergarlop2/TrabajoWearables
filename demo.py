import json
import numpy as np
from sense_hat import SenseHat
import torch
import time
from collections import deque


class OnlineActivityClassifier:
    def __init__(self, model_path, norm_values_path, window_size=80):
        # Load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

        # Load normalization values
        with open(norm_values_path, "r") as f:
            self.norm_values = json.load(f)

        # Initialize sensor
        self.sense = SenseHat()
        self.window_size = window_size

        # Initialize sliding windows for each sensor
        self.windows = {
            "pitch": deque(maxlen=window_size),
            "roll": deque(maxlen=window_size),
            "yaw": deque(maxlen=window_size),
            "x_accel": deque(maxlen=window_size),
            "y_accel": deque(maxlen=window_size),
            "z_accel": deque(maxlen=window_size),
        }

        # Activities mapping
        self.activities = ["baseball", "bolos", "boxeo", "golf", "tenis", "reposo"]

    def update_windows(self):
        """Update all sensor windows with new readings"""
        orientation = self.sense.get_orientation()
        acceleration = self.sense.get_accelerometer_raw()

        self.windows["pitch"].append(orientation["pitch"])
        self.windows["roll"].append(orientation["roll"])
        self.windows["yaw"].append(orientation["yaw"])
        self.windows["x_accel"].append(acceleration["x"])
        self.windows["y_accel"].append(acceleration["y"])
        self.windows["z_accel"].append(acceleration["z"])

    def get_normalized_data(self):
        """Convert current windows to normalized numpy array"""
        if len(self.windows["pitch"]) < self.window_size:
            return None

        features = []
        for i, key in enumerate(
            ["pitch", "roll", "yaw", "x_accel", "y_accel", "z_accel"]
        ):
            values = list(self.windows[key])
            normalized_values = (
                np.array(values) - self.norm_values["mean"][i]
            ) / self.norm_values["std"][i]
            features.append(normalized_values)

        return np.array(features)

    def predict(self):
        """Make prediction with current window of data"""
        data = self.get_normalized_data()
        if data is None:
            return None

        # Prepare data for model
        data = torch.FloatTensor(data).unsqueeze(0)  # Add batch dimension
        data = data.to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(data)
            prediction = torch.argmax(outputs, dim=1).item()

        return self.activities[prediction]

    def run_demo(self, duration=None, update_interval=0.5):
        """
        Run continuous demo for specified duration
        If duration is None, run indefinitely until KeyboardInterrupt
        """
        print("Starting activity recognition demo...")
        print("Press Ctrl+C to stop")

        try:
            start_time = time.time()
            while True:
                if duration and (time.time() - start_time) > duration:
                    break

                self.update_windows()
                prediction = self.predict()

                if prediction:
                    print(f"\rCurrent activity: {prediction}", end="", flush=True)

                time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\nDemo stopped by user")


if __name__ == "__main__":
    classifier = OnlineActivityClassifier(
        model_path="best_activity_model.pt",
        norm_values_path="normalization_values.json",
    )
    classifier.run_demo()
