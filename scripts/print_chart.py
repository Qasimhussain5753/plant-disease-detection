import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# -----------------------------
# Load the saved training history
# -----------------------------
with open('./models/history.pkl', 'rb') as f:
    history_data = pickle.load(f)

# -----------------------------
# Plot Accuracy and Loss Curves
# -----------------------------
def plot_accuracy_and_loss(history_data):
    epochs_range = range(1, len(history_data['accuracy']) + 1)
    plt.figure(figsize=(14, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_data['accuracy'], 'bo-', label='Train Acc')
    plt.plot(epochs_range, history_data['val_accuracy'], 'ro-', label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_data['loss'], 'bo-', label='Train Loss')
    plt.plot(epochs_range, history_data['val_loss'], 'ro-', label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()


# -----------------------------
# Show Example Predictions
# -----------------------------
def plot_example_predictions(data, predictions_probs, class_names, num_images=9):
    predictions = np.argmax(predictions_probs, axis=1)
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        img, label = data[i]
        true_label = class_names[np.argmax(label)]
        pred_label = class_names[predictions[i]]
        prob = np.max(predictions_probs[i])
        color = 'green' if true_label == pred_label else 'red'

        plt.subplot(3, 3, i + 1)
        plt.imshow(img.astype("float32"))
        plt.title(f"True: {true_label}\nPred: {pred_label} ({prob:.2f})", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Run All Visualizations
# -----------------------------
plot_accuracy_and_loss(history_data)

