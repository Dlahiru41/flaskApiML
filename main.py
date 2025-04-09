from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask_cors import CORS
import os
import tempfile
import requests
from io import BytesIO

app = Flask(__name__)
# Configure CORS properly to allow requests from your frontend
CORS(app, resources={r"/*": {"origins": "*"}})

# Update these paths to where you've stored your models
MODEL_DIR = 'C:\\Users\\dlahi\\Downloads\\'  # Change this to your actual model directory
FEATURE_EXTRACTOR_PATH = os.path.join(MODEL_DIR, 'feature_extractor.h5')
CLUSTERING_MODEL_PATH = os.path.join(MODEL_DIR, 'clustering_model.joblib')

# Load models once at startup for better performance
print("Loading models...")
feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
clustering_model = joblib.load(CLUSTERING_MODEL_PATH)
print("Models loaded successfully")


@app.route('/predict-from-url', methods=['POST'])
def predict_from_url():
    data = request.json

    if not data or 'imageUrl' not in data:
        return jsonify({'error': 'No image URL provided'}), 400

    image_url = data.get('imageUrl')
    filename = data.get('filename', 'unknown.jpg')

    try:
        # Download the image from Firebase
        print(f"Downloading image from: {image_url}")
        response = requests.get(image_url)

        if response.status_code != 200:
            return jsonify({'error': f'Failed to download image: {response.status_code}'}), 500

        # Load the image from the response content
        image_content = BytesIO(response.content)

        # Process image - resize and preprocess for EfficientNet
        img = load_img(image_content, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features using the feature extractor
        features = feature_extractor.predict(img_array, verbose=0)

        # Predict cluster using the clustering model
        pattern_id = int(clustering_model.predict(features)[0])

        # Calculate confidence based on distance to cluster center
        cluster_center = clustering_model.cluster_centers_[pattern_id]
        distance = np.linalg.norm(features.flatten() - cluster_center)

        # Convert distance to confidence (inverse relationship)
        max_distance = 10.0  # This threshold depends on your feature space
        confidence = float(max(0, 1 - (distance / max_distance)))

        print(f"Prediction: Pattern {pattern_id + 1}, Confidence: {confidence:.2f}")

        # Return the prediction results
        return jsonify({
            'pattern': pattern_id + 1,  # Make it 1-indexed (matching your clusters)
            'confidence': confidence
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Keep the original endpoint for direct file uploads if needed
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_file = request.files['image']

        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
            temp_path = temp.name
            image_file.save(temp_path)

        # Process image - resize and preprocess for EfficientNet
        img = load_img(temp_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features using the feature extractor
        features = feature_extractor.predict(img_array, verbose=0)

        # Predict cluster using the clustering model
        pattern_id = int(clustering_model.predict(features)[0])

        # Calculate confidence based on distance to cluster center
        cluster_center = clustering_model.cluster_centers_[pattern_id]
        distance = np.linalg.norm(features.flatten() - cluster_center)

        # Convert distance to confidence (inverse relationship)
        max_distance = 10.0
        confidence = float(max(0, 1 - (distance / max_distance)))

        # Clean up the temporary file
        os.unlink(temp_path)

        # Return the prediction results
        return jsonify({
            'pattern': pattern_id + 1,
            'confidence': confidence
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)