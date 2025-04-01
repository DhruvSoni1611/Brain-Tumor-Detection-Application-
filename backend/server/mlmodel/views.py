import tensorflow as tf
import numpy as np
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from PIL import Image
import io
import os
import gdown

# Define model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model.h5')

# Drive Link : https://drive.google.com/file/d/15a3Lg-Ch9gGHgPcqIUMDPp8XuSLYyUuN/view?usp=sharing
# Google Drive file ID
GDRIVE_FILE_ID = "15a3Lg-Ch9gGHgPcqIUMDPp8XuSLYyUuN"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model.h5 from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# Load the model
# model_path = os.path.join(os.path.dirname(__file__), '../model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Define tumor classes (Modify as per your dataset labels)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


@api_view(["POST"])
@parser_classes([MultiPartParser])
def predict_tumor(request):
    if "image" not in request.FILES:
        return JsonResponse({"error": "No image provided"}, status=400)

    # Read Image
    image = request.FILES["image"]
    img = Image.open(image).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    # Expand dimensions for batch
    img_array = np.expand_dims(img_array, axis=0)

    # Make Prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    return JsonResponse({"tumor_type": predicted_class, "confidence": confidence})
