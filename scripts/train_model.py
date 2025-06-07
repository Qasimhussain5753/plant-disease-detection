import os
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -----------------------------
# Load Configuration from YAML
# -----------------------------
with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Paths
dataset_path = config['paths']['dataset']
model_path = config['paths']['model']
history_path = config['paths']['history']
class_indices_path = config['paths']['class_indices']
checkpoint_dir = config['paths'].get('checkpoints', './checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# Parameters
img_size = tuple(config['image']['size'])
channels = config['image']['channels']
input_shape = img_size + (channels,)
batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
learning_rate = config['training']['learning_rate']
validation_split = config['training']['validation_split']
augment = config['augmentation']
resume_training = config['training'].get('resume', False)

# -----------------------------
# Data Preprocessing
# -----------------------------
datagen = ImageDataGenerator(
    rescale=augment['rescale'],
    validation_split=validation_split,
    rotation_range=augment['rotation_range'],
    zoom_range=augment['zoom_range'],
    horizontal_flip=augment['horizontal_flip']
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Save class indices
os.makedirs(os.path.dirname(class_indices_path), exist_ok=True)
with open(class_indices_path, 'wb') as f:
    pickle.dump(train_data.class_indices, f)

# -----------------------------
# Load EfficientNetB0
# -----------------------------
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# Resume from Checkpoint
# -----------------------------
initial_epoch = 0
# Find best model file
# Find best model file
# Resume from latest .weights.h5 file
if resume_training:
    weight_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.weights.h5')]
    if weight_files:
        latest = max(weight_files, key=lambda x: int(x.split('-epoch-')[1].split('.')[0]))
        initial_epoch = int(latest.split('-epoch-')[1].split('.')[0])
        model.load_weights(os.path.join(checkpoint_dir, latest))
        print(f"Resuming from checkpoint {latest} at epoch {initial_epoch}")
    else:
        print("No checkpoint found. Training from scratch.")

# -----------------------------
# Define Checkpoint Callback
# -----------------------------
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "model-epoch-{epoch:02d}.weights.h5"),
    save_weights_only=True,
    save_freq='epoch',
    save_best_only=False,
    verbose=1
)

# -----------------------------
# Train the Model
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    initial_epoch=initial_epoch,
    callbacks=[checkpoint_callback]
)

# -----------------------------
# Save Model & History
# -----------------------------
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)

# -----------------------------
# Evaluation & Report
# -----------------------------
val_data.reset()
y_pred_probs = model.predict(val_data)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_data.classes
class_names = list(val_data.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# -----------------------------
# Plot Accuracy and Loss Curves
# -----------------------------
def plot_accuracy_and_loss(history_data):
    epochs_range = range(1, len(history_data['accuracy']) + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_data['accuracy'], 'bo-', label='Train Acc')
    plt.plot(epochs_range, history_data['val_accuracy'], 'ro-', label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

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

plot_accuracy_and_loss(history.history)

# -----------------------------
# Confusion Matrix
# -----------------------------
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_true, y_pred, class_names)

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

val_data.reset()
example_images, _ = next(val_data)
plot_example_predictions(example_images, y_pred_probs[:9], class_names)
