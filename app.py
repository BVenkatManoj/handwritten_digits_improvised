import base64
import io
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load your trained .keras model
try:
    model = tf.keras.models.load_model('handwritten_digits.keras')
    print("🚀 Model loaded successfully with advanced telemetry!")
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
        
        # Preprocess image
        image = image.convert('L') 
        image = image.resize((28, 28)) 
        image = image.point(lambda p: p if p > 30 else 0)
        
        # Normalize and reshape
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 784)
        
        # Predict & handle logits via Softmax
        raw_logits = model.predict(img_array)
        probabilities = tf.nn.softmax(raw_logits).numpy()[0] # Grab first batch item array
        
        # Find primary prediction
        predicted_digit = int(np.argmax(probabilities))
        
        # Feature 1: Extract the top 3 predictions with confidence levels
        top_indices = np.argsort(probabilities)[-3:][::-1] # Sorts ascending, takes last 3, reverses
        top_predictions = [
            {"digit": int(idx), "confidence": round(float(probabilities[idx]) * 100, 2)}
            for idx in top_indices
        ]
        
        return jsonify({
            'prediction': predicted_digit,
            'top_predictions': top_predictions
        })
        
    except Exception as e:
        print(f"\n❌ ERROR DURING PREDICTION: {str(e)}\n")
        return jsonify({'error': str(e)}), 500

# Feature 2: Route to log feedback metrics (for tracking model accuracy over time)
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    is_correct = data.get('correct')
    user_digit = data.get('actualDigit')
    
    # In a real production app, you would append this to a database or file
    print(f"📊 USER FEEDBACK RECEIVED: Was Correct? {is_correct} | True Digit: {user_digit}")
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)