import os
import gc
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add your model directory to sys.path if needed
sys.path.append('dfdc_model')

from dfdc_model.kernel_utils import VideoReader, FaceExtractor
from dfdc_model.training.zoo.classifiers import DeepFakeClassifier
from enhanced_prediction import enhanced_predict_on_video

real_dir = r"C:\Users\babel\Downloads\archive\FaceForensics++_C23\original"
fake_dir = r"C:\Users\babel\Downloads\archive\FaceForensics++_C23\FaceSwap"

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_models():
    models = []
    weights_dir = "dfdc_model/weights"
    model_files = [
        "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40",
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model_file in model_files:
        model_path = os.path.join(weights_dir, model_file)
        if os.path.exists(model_path):
            model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to(device)
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=True)
            model.eval()
            del checkpoint
            if device == "cuda":
                models.append(model.half())
            else:
                models.append(model)
    return models

def initialize_preprocessing():
    frames_per_video = 10  # Process 5 frames per video
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)
    input_size = 380
    return face_extractor, input_size, frames_per_video

def smart_prediction_strategy(predictions):
    predictions = np.array(predictions)
    if len(predictions) == 0:
        return 0.5
    fake_frames_50 = np.sum(predictions > 0.5)
    fake_frames_60 = np.sum(predictions > 0.6)
    fake_frames_70 = np.sum(predictions > 0.7)
    total_frames = len(predictions)
    fake_ratio = fake_frames_50 / total_frames
    if fake_ratio > 0.5:
        if fake_frames_70 > total_frames * 0.3:
            return min(0.95, np.mean(predictions[predictions > 0.7]))
        elif fake_frames_60 > total_frames * 0.4:
            return min(0.9, np.mean(predictions[predictions > 0.6]))
        else:
            suspicious_mean = np.mean(predictions[predictions > 0.5])
            overall_mean = np.mean(predictions)
            return (suspicious_mean * 0.7 + overall_mean * 0.3)
    elif fake_ratio > 0.3:
        return min(0.7, np.mean(predictions))
    else:
        return np.mean(predictions)

def evaluate(dataset_csv, real_dir, fake_dir):
    df = pd.read_csv(dataset_csv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models()
    face_extractor, input_size, frames_per_video = initialize_preprocessing()

    y_trues = []
    y_preds = []

    loop = tqdm(df.iterrows(), total=len(df), desc="Evaluating")
    for idx, row in loop:
        # Determine source dir
        filename = row["filename"]
        if row["label"] == 0:
            file_path = os.path.join(real_dir, filename)
        else:
            file_path = os.path.join(fake_dir, filename)
        gt_label = row["label"]

        try:
            result = enhanced_predict_on_video(
                face_extractor=face_extractor,
                video_path=file_path,
                batch_size=1,  # Keep batch size at 1 for memory efficiency
                input_size=input_size,
                models=models,
                strategy=smart_prediction_strategy,
                apply_compression=False,
                return_frame_predictions=False
            )
            pred_score = result["prediction"]
            pred_label = 1 if pred_score > 0.5 else 0
            y_trues.append(gt_label)
            y_preds.append(pred_label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            y_trues.append(gt_label)
            y_preds.append(0.5)  # neutral

        cleanup_memory()

    # Metrics
    y_trues_np = np.array(y_trues)
    y_preds_np = np.array(y_preds)
    correct = y_preds_np != 0.5  # exclude neutral (errors)
    acc = accuracy_score(y_trues_np[correct], y_preds_np[correct])
    prec = precision_score(y_trues_np[correct], y_preds_np[correct])
    rec = recall_score(y_trues_np[correct], y_preds_np[correct])
    cm = confusion_matrix(y_trues_np[correct], y_preds_np[correct])

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)

        # Plot confusion matrix and metrics, save as PNG
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

        # Add metrics as text box
    metrics_text = f"Accuracy: {acc:.2f}\nPrecision: {prec:.2f}\nRecall: {rec:.2f}"
    plt.gcf().text(0.7, 0.2, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
        "total": len(df),
        "evaluated": sum(correct)
    }

if __name__ == "__main__":
    # Usage
    # Make sure 'evaluation_list.csv' is created as above
    dataset_csv = "evaluation_list.csv"
    stats = evaluate(dataset_csv, real_dir, fake_dir)
    print(stats)
