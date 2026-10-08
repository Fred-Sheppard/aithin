import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class Metrics:
    def __init__(self, model_path, csv_path, image_dir, img_size=(128, 128)):
        self.model = load_model(model_path)
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.img_size = img_size
        self.X_test, self.y_test, self.label_to_letter = self.load_test_data()

    def load_test_data(self):
        # Load CSV file
        df = pd.read_csv(self.csv_path)

        # Map letters to labels
        unique_letters = df["img_letter"].unique()
        letter_to_label = {letter: i for i, letter in enumerate(unique_letters)}
        label_to_letter = {i: letter for letter, i in letter_to_label.items()}
        df["label"] = df["img_letter"].map(letter_to_label)

        # Load images and labels
        images, labels = [], []
        for _, row in df.iterrows():
            img_path = os.path.join(self.image_dir, f"{row['img_no']:05}.png")
            if os.path.exists(img_path):
                img = load_img(img_path, color_mode="grayscale", target_size=self.img_size)
                img = img_to_array(img) / 255.0  # Normalize
                images.append(img)
                labels.append(row["label"])

        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        # One-hot encode labels
        labels = to_categorical(labels, num_classes=len(unique_letters))

        return images, labels, label_to_letter

    def evaluate_model(self):
        y_pred_probs = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        # Each image represents one Ogham letter, so CER is the fraction of images whose predicted letter differs from the reference letter
        character_error_rate = np.mean(y_pred != y_true)
        print(f"Character Error Rate (CER): {character_error_rate:.4f} ({character_error_rate * 100:.2f}%)")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.label_to_letter.values(), yticklabels=self.label_to_letter.values())
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

        # Classification Report
        report = classification_report(y_true, y_pred, digits=4, target_names=self.label_to_letter.values())
        print("Classification Report:\n", report)

# Example Usage:
metrics = Metrics("best_ogham_hybrid_model.h5", "output/labels.csv", "output/temp-letters-dir")
metrics.evaluate_model()