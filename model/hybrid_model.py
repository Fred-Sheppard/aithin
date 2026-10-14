import os
# Suppress TensorFlow and CUDA warnings/logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom operations
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"  # Suppress verbose logging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Suppress CUDA errors

import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import json
import glob
from PIL import Image, ImageOps
from image_preprocessing import preprocess_image


def load_image(image_path):
    """Grayscale image, upright per its EXIF orientation, with transparency composited onto white."""
    img = ImageOps.exif_transpose(Image.open(image_path))
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    return img.convert("L")


def preprocess(image, target_size=(128, 128)):
    """Model input for a grayscale PIL image: float32 array shaped (height, width, 1) in [0, 1]."""
    return preprocess_image(image, target_size=target_size)

def train_hybrid_model(model_name, csv_path=os.getcwd() + "/output/labels.csv", image_dir=os.getcwd() + "/output/temp-letters-dir"):
    # Load the CSV file
    print(os.getcwd())
    df = pd.read_csv(csv_path)

    # Map 'img_letter' to numerical labels
    unique_letters = df["img_letter"].unique()
    letter_to_label = {letter: i for i, letter in enumerate(unique_letters)}
    label_to_letter = {i: letter for letter, i in letter_to_label.items()}

    df["label"] = df["img_letter"].map(letter_to_label)

    # Load images and labels
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, f"{row['img_no']:05}.png")
        if os.path.exists(img_path):
            img = preprocess(load_image(img_path))
            images.append(img)
            labels.append(row["label"])

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # One-hot encode labels
    labels = to_categorical(labels, num_classes=len(unique_letters))

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Define the hybrid CNN-RNN model
    input_shape = (128, 128, 1)

    # CNN Component
    inputs = Input(shape=input_shape)

    # CNN layers for feature extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Reshape for RNN
    # After 3 max pooling layers (2x2), the 128x128 image becomes (128/8)x(128/8)x128 = 16x16x128
    x = Reshape((16, 16*128))(x)

    # RNN layers
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64))(x)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(unique_letters), activation='softmax')(x)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Add callbacks for better training
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint(model_name, save_best_only=True)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # We'll use early stopping to determine the actual number of epochs
        batch_size=32,
        callbacks=callbacks
    )

    # # Plot training history
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='lower right')

    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper right')

    # plt.tight_layout()
    # plt.savefig('training_history.png')

    # # Evaluate the model
    # model = load_model(model_name)  # Load the best model saved by ModelCheckpoint
    # loss, accuracy = model.evaluate(X_val, y_val)
    # print(f"Validation Loss: {loss}")
    # print(f"Validation Accuracy: {accuracy}")

    # # Save the mapping for prediction
    # with open('label_to_letter.json', 'w') as f:
    #     json.dump({str(k): v for k, v in label_to_letter.items()}, f)

    return model, label_to_letter


# Predict on a new image
def predict_image(model_path, image_path, label_to_letter=os.path.dirname(os.path.abspath(__file__)) + "/label_to_letter.json"):
    print(label_to_letter)
    if (not os.path.exists(model_path)):
        print("Model not found. Please enter valid model path or train the model first.")
        exit(1)
    model = load_model(model_path)

    # Load label mapping
    with open(label_to_letter, 'r') as f:
        label_to_letter = json.load(f)
        label_to_letter = {int(k): v for k, v in label_to_letter.items()}

    # Preprocess the image
    img = preprocess(load_image(image_path))
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    confidence = prediction[0][predicted_label]

    # Get top 3 predictions for ambiguous cases
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    top_predictions = [(label_to_letter[idx], float(prediction[0][idx])) for idx in top_indices]

    # plt.imshow(img[0, :, :, 0], cmap="gray")
    # plt.title(f"Predicted: {label_to_letter[predicted_label]} ({confidence*100:.2f}%)")
    # plt.axis("off")
    # plt.show()


    return {
        'predicted': label_to_letter[predicted_label],
        'confidence': float(confidence),
        'top_predictions': top_predictions
    }

# Function to run inference on a set of images
@DeprecationWarning
def batch_predict(image_dir, pattern="*.png"):
    model = load_model("best_ogham_hybrid_model.h5")
    results = []

    for img_path in glob.glob(os.path.join(image_dir, pattern)):
        result = predict_image(img_path, model)
        results.append({
            'image': os.path.basename(img_path),
            'prediction': result
        })

    return results
def generate_heatmap(model, img_array, pred_index=None):
    """
    Generate a heatmap using Grad-CAM technique
    
    Args:
        model: Pre-trained model
        img_array: Preprocessed image array
        pred_index: Index of the predicted class (if None, uses the class with highest score)
    
    Returns:
        Heatmap overlaid on the original image
    """
    # Expand dimensions to match model input requirements
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Get the last convolutional layer
    # (You may need to adjust this based on your model architecture)
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        raise ValueError("No convolutional layer found in the model")
    
    # Create a model that maps the input image to the activations of the last conv layer
    last_conv_layer_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )
    
    # Create a model that maps the activations of the last conv layer to the final class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[model.layers.index(last_conv_layer) + 1:]:
        x = layer(x)
    classifier_model = tf.keras.models.Model(classifier_input, x)
    
    # Compute the gradient of the top predicted class with respect to the output feature map
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_tensor)
        tape.watch(last_conv_layer_output)
        
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradient of the predicted class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by corresponding gradients
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    # Weight each channel in the feature map array by importance towards the predicted class
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    # Average over all channels to get heatmap
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    
    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    # Resize heatmap to match original image dimensions
    original_img_shape = img_array.shape[:2]
    heatmap = cv2.resize(heatmap, (original_img_shape[1], original_img_shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on original image
    img_array_rgb = np.uint8(255 * img_array)
    if len(img_array_rgb.shape) == 2:  # If grayscale, convert to RGB
        img_array_rgb = cv2.cvtColor(img_array_rgb, cv2.COLOR_GRAY2RGB)
    
    superimposed_img = heatmap * 0.4 + img_array_rgb
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    return superimposed_img, heatmap

def visualize_ogham_attention(model_path, image_path, letter_classes=None, output_path=None):
    """
    Load an Ogham image, predict its class, and visualize the model's attention
    
    Args:
        model_path: Path to the saved model
        image_path: Path to the Ogham letter image
        letter_classes: List of Ogham letter classes in prediction order
        output_path: Path to save the visualization (if None, will display instead)
    """
    # Load the model
    model = load_model(model_path)
    
    # Load and preprocess the image
    input_shape = model.input_shape[1:3]  # (height, width)
    img_array = preprocess_image(image_path, target_size=input_shape)
    
    # Make prediction
    pred = model.predict(np.expand_dims(img_array, axis=0))
    pred_class = np.argmax(pred[0])
    confidence = pred[0][pred_class]
    
    # Get class name if available
    label_to_letter = os.path.dirname(os.path.abspath(__file__)) + "/label_to_letter.json"
    with open(label_to_letter, 'r') as f:
        label_to_letter = json.load(f)
        label_to_letter = {int(k): v for k, v in label_to_letter.items()}

    class_name = label_to_letter[pred_class]
    if letter_classes and pred_class < len(letter_classes):
        class_name = letter_classes[pred_class]
    
    # Generate heatmap
    superimposed_img, raw_heatmap = generate_heatmap(model, img_array, pred_class)
    
    # Plot the results
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_array[:, :, 0], cmap='gray')
    axes[0].set_title('Original Ogham Letter')
    axes[0].axis('off')
    
    
    # Superimposed
    axes[1].imshow(superimposed_img)
    axes[1].set_title(f'Predicted: {class_name} ({confidence:.2f})')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    return superimposed_img
