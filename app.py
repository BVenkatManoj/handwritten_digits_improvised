import base64
import io
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load your trained .keras model
model = None
try:
    model = tf.keras.models.load_model('handwritten_digits.keras')
    print("🚀 Model loaded successfully!")
    # Print internal model expected shape for diagnosis
    if hasattr(model, 'input_shape'):
        print(f"📊 Model expects input shape internally: {model.input_shape}")
except Exception as e:
    print(f"❌ Error loading model file: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data found in request'}), 400
            
        image_data = data['image']
        
        # Decode base64 image
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image to standard 28x28 grayscale
        image = image.convert('L') 
        image = image.resize((28, 28)) 
        image = image.point(lambda p: p if p > 30 else 0)
        
        # Normalize the pixel data
        img_array = np.array(image) / 255.0
        
        # Determine the target layer shape directly from what the model is demanding
        expected_shape = model.input_shape if hasattr(model, 'input_shape') else None
        
        # Dynamic fallback: Morph tensor layout to match the layer rules exactly
        if expected_shape and len(expected_shape) == 2 and expected_shape[1] == 784:
            # Model needs flat layout
            img_array = img_array.reshape(1, 784)
        else:
            # Model needs 4D image grid grid layout
            img_array = img_array.reshape(1, 28, 28, 1)
            
        # Convert to Tensor object
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        print(f"📡 Sending shape {img_tensor.shape} directly to model functional layer layers...")
        
        # Invoke model direct call bypass to avoid predict() wrapper graph state conflicts
        raw_logits = model(img_tensor, training=False)
        
        # Apply Softmax activation manually to ensure raw values are valid probabilities
        probabilities = tf.nn.softmax(raw_logits).numpy()[0]
        
        # Find primary prediction
        predicted_digit = int(np.argmax(probabilities))
        
        # Extract top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [
            {"digit": int(idx), "confidence": round(float(probabilities[idx]) * 100, 2)}
            for idx in top_indices
        ]
        
        return jsonify({
            'prediction': predicted_digit,
            'top_predictions': top_predictions
        })
        
    except Exception as e:
        print(f"\n❌ ERROR DURING INFERENCE PIPELINE: {str(e)}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    print(f"📊 USER FEEDBACK RECEIVED: {data}")
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)