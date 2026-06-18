# 🧠 Handwritten Digit Recognition Web Application Using TensorFlow, CNN & Flask

A full-stack deep learning application that recognizes handwritten digits (0–9) in real time using a Convolutional Neural Network (CNN). The project combines TensorFlow/Keras for model development, Flask for backend deployment, and HTML5 Canvas with JavaScript for an interactive web interface.

The application allows users to draw digits directly in their browser and receive instant predictions along with confidence scores. It demonstrates the complete machine learning workflow, from data preprocessing and model training to deployment as a real-world web application.

---

# 🚀 Overview

This project was developed to classify handwritten digits using the MNIST dataset. A CNN model is trained to recognize digit patterns and is then deployed through a Flask web server. Users interact with an HTML5 Canvas, draw digits, and receive real-time predictions without refreshing the page.

The project showcases the integration of:

* Deep Learning
* Computer Vision
* Web Development
* Image Processing
* Machine Learning Deployment

into a single end-to-end AI application.

---

# ✨ Features

* Handwritten digit recognition (0–9)
* Convolutional Neural Network (CNN)
* Interactive HTML5 drawing canvas
* Real-time predictions
* TensorFlow/Keras model deployment
* Flask REST API backend
* Image preprocessing using Pillow (PIL)
* Softmax confidence scoring
* Top-3 prediction probabilities
* Training history visualization
* Confusion matrix analysis
* Responsive browser interface
* Model export and deployment support

---

# 📂 Project Structure

```plaintext
digit-recognizer/
│
├── handwritten_digits_model.ipynb
├── app.py
├── handwritten_digits.keras
├── requirements.txt
│
└── templates/
    └── index.html
```

---

## 📓 handwritten_digits_model.ipynb

This notebook contains the complete machine learning workflow used to build, train, evaluate, and export the CNN model.

### Responsibilities

* Load the MNIST dataset
* Preprocess and normalize image data
* Visualize sample images
* Build the CNN architecture
* Train and validate the model
* Generate predictions
* Plot training metrics
* Create confusion matrices
* Evaluate model performance
* Export the trained model as `.keras`

The notebook serves as the experimentation and development environment before deployment.

---

## 🐍 app.py

The Flask backend acts as the bridge between the browser interface and the trained CNN model.

### Responsibilities

* Load the trained model
* Host application routes
* Receive image data from the frontend
* Execute preprocessing operations
* Run model inference
* Apply Softmax probability conversion
* Return prediction results in JSON format

---

## 🧠 handwritten_digits.keras

This file stores the trained TensorFlow/Keras CNN model, including:

* Model architecture
* Learned weights
* Bias parameters
* Layer configurations
* Training parameters

The model is loaded directly by Flask during application startup.

---

## 📦 requirements.txt

Contains all project dependencies:

```txt
flask
numpy
pillow
tensorflow
matplotlib
scikit-learn
```

Installation:

```bash
pip install -r requirements.txt
```

---

## 🌐 templates/index.html

The frontend interface is built using:

* HTML5
* CSS3
* JavaScript

### Features

* Interactive drawing canvas
* Mouse support
* Touchscreen support
* Real-time prediction updates
* Confidence score display
* Clear canvas functionality
* Responsive design

---

# 📁 Dataset

The model is trained using the MNIST Handwritten Digit Dataset.

MNIST is one of the most widely used benchmark datasets in machine learning and computer vision.

## Dataset Statistics

| Property        | Value          |
| --------------- | -------------- |
| Total Images    | 70,000         |
| Training Images | 60,000         |
| Testing Images  | 10,000         |
| Classes         | 10 (0–9)       |
| Image Size      | 28 × 28 Pixels |
| Color Format    | Grayscale      |

Each image contains a handwritten digit represented by grayscale pixel values ranging from 0 to 255.

---

# 🔄 Data Preprocessing

Before training, the dataset undergoes several preprocessing steps.

### 1. Normalization

Pixel values are scaled from:

```text
0–255
```

to:

```text
0–1
```

This improves training stability and convergence.

### 2. Reshaping

Images are reshaped from:

```text
28 × 28
```

to:

```text
28 × 28 × 1
```

to include the grayscale channel required by CNN layers.

### 3. Data Visualization

Random samples are displayed to verify:

* Label correctness
* Image quality
* Dataset consistency

---

# 🏗️ Model Architecture

The model is implemented using a Convolutional Neural Network (CNN), which is highly effective for image classification tasks because it automatically learns visual features such as edges, curves, and digit shapes.

## CNN Architecture

| Layer                    | Output Shape | Activation |
| ------------------------ | ------------ | ---------- |
| Input Layer              | 28 × 28 × 1  | —          |
| Conv2D (32 Filters, 3×3) | 26 × 26 × 32 | ReLU       |
| MaxPooling2D (2×2)       | 13 × 13 × 32 | —          |
| Conv2D (64 Filters, 3×3) | 11 × 11 × 64 | ReLU       |
| MaxPooling2D (2×2)       | 5 × 5 × 64   | —          |
| Flatten                  | 1600         | —          |
| Dense                    | 128          | ReLU       |
| Output Layer             | 10           | Softmax    |

## Architecture Flow

```plaintext
Input Image (28×28×1)
          │
          ▼
Conv2D (32 Filters)
          │
          ▼
MaxPooling2D
          │
          ▼
Conv2D (64 Filters)
          │
          ▼
MaxPooling2D
          │
          ▼
Flatten
          │
          ▼
Dense (128 ReLU)
          │
          ▼
Output Layer (10 Softmax)
```

### Why CNN?

* Preserves spatial information.
* Learns important image features automatically.
* Reduces overfitting through parameter sharing.
* Achieves significantly higher accuracy than standard dense networks.
* Widely used in computer vision applications.

---

# ⚙️ Training Configuration

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Optimizer     | Adam                            |
| Learning Rate | 0.001                           |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size    | 32                              |
| Epochs        | 10–20                           |
| Classes       | 10                              |

### Why Adam?

* Adaptive learning rates
* Fast convergence
* Excellent performance on image classification tasks

---

# 📈 Model Training

During training, the CNN learns hierarchical image features.

### Feature Learning Process

**Early Layers**

* Detect edges
* Detect lines
* Detect simple shapes

**Deeper Layers**

* Detect digit strokes
* Learn curves and loops
* Recognize complete digit structures

Training progress is monitored using:

* Accuracy
* Loss

---

# 📊 Evaluation & Visualization

Several evaluation techniques are used.

## Training Curves

### Accuracy Curve

Shows improvement in prediction accuracy across epochs.

### Loss Curve

Shows reduction in classification error during training.

---

## Sample Predictions

Random MNIST samples are displayed with:

* Actual label
* Predicted label
* Confidence score

---

## Confusion Matrix

A confusion matrix is generated to analyze:

![Confusion Matrix](images/confusion_matrix.png)
* Correct classifications
* Misclassifications
* Per-class performance

This provides a detailed understanding of the model's strengths and weaknesses.

---

# 🌐 Frontend Architecture

The frontend is built using HTML5 Canvas, CSS, and JavaScript.

## HTML5 Canvas

Acts as a digital drawing board where users can draw digits using:

* Mouse input
* Touch input

Canvas Size:

```text
280 × 280 Pixels
```

---

## JavaScript Processing

When the Predict button is clicked:

1. Canvas image is captured.
2. Image is converted into Base64 format.
3. Data is packaged into JSON.
4. Fetch API sends the request to Flask.

All communication is asynchronous, meaning the page never reloads.

---

# 🐍 Backend Processing Pipeline

The Flask server receives the image and performs preprocessing before inference.

## Step 1: Decode Image

Convert Base64 image data into raw image bytes.

## Step 2: Convert to Grayscale

Remove unnecessary color channels.

## Step 3: Resize

Resize image from:

```text
280 × 280
```

to:

```text
28 × 28
```

to match MNIST dimensions.

## Step 4: Thresholding

Remove anti-aliasing artifacts and improve image clarity.

## Step 5: Normalize

Scale pixel values between:

```text
0.0 – 1.0
```

## Step 6: Reshape

Convert image into:

```text
(1, 28, 28, 1)
```

which matches the CNN input shape.

---

# 🧠 Prediction Pipeline

The processed image is passed into the trained CNN model.

The output layer generates probabilities for each digit class.

Example:

```text
Digit 0: 0.01%
Digit 1: 0.03%
Digit 2: 0.12%
...
Digit 8: 98.45%
...
```

The digit with the highest probability is selected as the final prediction.

```python
prediction = np.argmax(probabilities)
```

The application returns:

* Predicted digit
* Confidence score
* Top-3 candidate predictions

---

# 🔄 End-to-End Workflow

```plaintext
User Draws Digit
        │
        ▼
HTML5 Canvas
        │
        ▼
JavaScript
(Base64 Encoding)
        │
        ▼
Fetch API Request
        │
        ▼
Flask Backend
(app.py)
        │
        ▼
Image Preprocessing
(PIL)
        │
        ▼
CNN Model
(.keras)
        │
        ▼
Probability Generation
        │
        ▼
Prediction
        │
        ▼
Results Displayed
In Browser
```

---

# 🛠️ Technologies Used

| Category                | Technology                      |
| ----------------------- | ------------------------------- |
| Programming Language    | Python                          |
| Deep Learning           | TensorFlow / Keras              |
| Neural Network          | CNN                             |
| Backend Framework       | Flask                           |
| Frontend                | HTML5, CSS3, JavaScript         |
| Image Processing        | Pillow (PIL)                    |
| Numerical Computing     | NumPy                           |
| Visualization           | Matplotlib                      |
| Evaluation              | Scikit-learn                    |
| Development Environment | Jupyter Notebook / Google Colab |

---

# 📥 Getting the Project

Clone the repository to your local machine using Git:

```bash
git clone https://github.com/your-username/digit-recognizer.git
```

Navigate to the project directory:

```bash
cd digit-recognizer
```

---

# ⚙️ Installation & Setup

### 1. Create a Virtual Environment (Recommended)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Install Dependencies

Install all required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 3. Verify Project Structure

Ensure the project directory contains:

```plaintext
digit-recognizer/
│
├── handwritten_digits_model.ipynb
├── app.py
├── handwritten_digits.keras
├── requirements.txt
│
└── templates/
    └── index.html
```

---

### 4. Launch the Flask Application

Start the backend server:

```bash
python app.py
```

If the application starts successfully, you should see output similar to:

```plaintext
* Running on http://127.0.0.1:5000
```

---

### 5. Open the Web Application

Open your browser and visit:

```plaintext
http://127.0.0.1:5000
```

You can now draw handwritten digits on the canvas and receive real-time predictions from the trained CNN model.

---

# 🧪 Running the Training Notebook

If you want to retrain the model:

1. Open Jupyter Notebook:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

2. Open:

```plaintext
handwritten_digits_model.ipynb
```

3. Run all cells to:

   * Load and preprocess the MNIST dataset
   * Train the CNN model
   * Evaluate performance
   * Generate visualizations
   * Export a new `handwritten_digits.keras` model


# 🎓 Key Learning Outcomes

* Convolutional Neural Networks (CNNs)
* Image classification
* TensorFlow model development
* Flask API deployment
* Image preprocessing
* Frontend-backend communication
* Real-time inference systems
* Model evaluation and visualization
* End-to-end AI deployment

---

# 🔮 Future Enhancements

### 1. Data Augmentation

Improve robustness using:

* Rotation
* Scaling
* Translation
* Noise injection

### 2. Active Learning System

Collect user corrections and retrain the model.

### 3. Cloud Deployment

Deploy using:

* Render
* Railway
* AWS
* Google Cloud
* Azure

### 4. Explainable AI

Integrate:

* Grad-CAM
* Saliency Maps
* Feature Visualizations

### 5. Mobile Optimization

Improve user experience on touch devices.

---

# 📝 Conclusion

This project demonstrates the complete lifecycle of a modern AI application, from training a CNN on the MNIST dataset to deploying it as an interactive web application using Flask. By combining deep learning, image processing, backend APIs, and frontend technologies, the system provides a practical example of real-world machine learning deployment and serves as a strong foundation for exploring advanced computer vision and deep learning techniques.

## 👨‍💻 Author

**B. Venkat Manoj**

Machine Learning & Deep Learning Enthusiast

Passionate about Artificial Intelligence, Machine Learning, Deep Learning, and Full-Stack AI Applications.
