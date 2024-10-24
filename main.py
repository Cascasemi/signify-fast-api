from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import cv2

# Initialize FastAPI app
app = FastAPI()

# Load the .h5 model using Keras' load_model method
model_path = 'model/model.h5'  # Path to your .h5 model
model = tf.keras.models.load_model(model_path)

# Define custom labels (these should match your model's output)
label_names = ['hello', 'iloveyou', 'thanks', 'yes', 'no']

# Route for detecting objects in an uploaded image
@app.post('/detect')
async def detect_objects(image: UploadFile = File(...)):
    try:
        # Read image data from the uploaded file
        contents = await image.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Preprocess the image (resize, normalize, etc.)
        img_resized = cv2.resize(img, (224, 224))  # Resize image to match your model's input size
        img_preprocessed = np.expand_dims(img_resized, axis=0)  # Add batch dimension
        img_preprocessed = img_preprocessed / 255.0  # Normalize pixel values to [0, 1]

        # Make prediction
        predictions = model.predict(img_preprocessed)
        predicted_class = np.argmax(predictions[0])  # Get the index of the highest probability

        # Get the label for the predicted class
        predicted_label = label_names[predicted_class]

        return JSONResponse(content={'class': predicted_label, 'confidence': str(predictions[0][predicted_class])})

    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)