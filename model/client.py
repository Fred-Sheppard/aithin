import os
# Suppress TensorFlow and CUDA warnings/logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom operations
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"  # Suppress verbose logging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Suppress CUDA errors

import argparse

def predict_convnext(args):
    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} not found. Train it with model/train.py and model/adapt.py.")
        exit(1)
    if args.train or args.heatmap:
        print("Error: --train and --heatmap are only supported for the Keras model.")
        exit(1)
    from ogham_convnext import predict_image
    result = predict_image(args.model, args.image)
    print(f"Predicted Letter: {result['predicted']} ({result['confidence'] * 100:.1f}%)")
    print("Top 3: " + " | ".join(f"{letter} ({p * 100:.1f}%)" for letter, p in result["top_predictions"]))


def main():
    parser = argparse.ArgumentParser(description="Image Classification Script")
    
    # Required arguments
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file (.pt for ConvNeXt, otherwise Keras)")
    parser.add_argument("--labels", type=str, default="model/label_to_letter.json", help="Path to the label_to_letter.json file")
    
    # Optional flag to train if model doesn't exist
    parser.add_argument("--train", action="store_true", help="Train the model if it does not exist")
    parser.add_argument("--heatmap", action="store_true", help="Generate heatmap visualization")
    args = parser.parse_args()

    if args.model.endswith(".pt"):
        predict_convnext(args)
        return

    # TensorFlow is only needed for the Keras model
    from hybrid_model import train_hybrid_model, predict_image, visualize_ogham_attention

    # Check if model exists, train if necessary
    if not os.path.exists(args.model):
        if args.train:
            print("Model not found. Training new model...")
            train_hybrid_model(args.model)  # Call training function
        else:
            print(f"Error: Model {args.model} not found. Use --train to create it.")
            exit(1)

    # Check if labels file exists
    if not os.path.exists(args.labels):
        print(f"Error: Labels file {args.labels} not found.")
        exit(1)
            
    if args.heatmap:
        visualize_ogham_attention(args.model, args.image)
    else:
        predicted_letter = predict_image(args.model, args.image, args.labels)
        print(f"Predicted Letter: {predicted_letter}")

if __name__ == "__main__":
    main()
