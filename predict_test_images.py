import pickle
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('models/model.h5')

# Load class_indices
with open('models/class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)

# Reverse it
idx_to_class = {v: k for k, v in class_indices.items()}

# Relative path to the folder
folder_path = Path("test_data")  # Or Path("./data")

for file_path in folder_path.iterdir():
    if file_path.is_file():  # Check if it's a file
        # Load and preprocess a single image
        print("")
        print(file_path)
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        print("Predicted class:", idx_to_class[predicted_class])

