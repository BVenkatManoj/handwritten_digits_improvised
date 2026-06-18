# 🖊️ Handwritten Digit Recognition Web Application Using TensorFlow & Flask

A full-stack deep learning application that recognizes handwritten digits drawn by users in real time. The project combines a TensorFlow/Keras neural network with a Flask backend and an interactive HTML5 Canvas frontend, demonstrating the complete workflow from model training to web deployment.

## 🚀 Overview

This project transforms a trained handwritten digit classification model into a browser-based application. Users can draw digits on a digital canvas, submit them for prediction, and instantly receive the model's predicted digit along with confidence scores.

The system bridges machine learning and web development by integrating TensorFlow, Flask, JavaScript, and image preprocessing techniques into a seamless client-server architecture.

## ✨ Features

* Interactive HTML5 drawing canvas
* Real-time digit prediction
* Flask-powered REST API
* Automatic image preprocessing pipeline
* Softmax probability visualization
* Top-3 prediction confidence scores
* Model evaluation and visualization
* Trained model export for deployment

## 📁 Dataset

The model was trained using the **MNIST Handwritten Digit Dataset**.

| Property           | Value          |
| ------------------ | -------------- |
| Total Images       | 70,000         |
| Classes            | Digits 0–9     |
| Image Size         | 28 × 28 Pixels |
| Color Format       | Grayscale      |
| Features per Image | 784            |

### Data Preprocessing

* Image normalization (0–255 → 0–1)
* Flattening 28×28 images into 784-feature vectors
* Dataset visualization and verification
* Preparation for neural network training

## 🏗️ Model Architecture

Built using TensorFlow/Keras Sequential API.

| Layer         | Units | Activation      |
| ------------- | ----- | --------------- |
| Input         | 784   | —               |
| Dense Layer 1 | 128   | ReLU            |
| Dense Layer 2 | 64    | ReLU            |
| Output Layer  | 10    | Linear (Logits) |

### Why This Architecture?

**ReLU**

* Introduces non-linearity
* Efficient training
* Helps reduce vanishing gradients

**Linear Output**

* Produces raw logits
* Compatible with Sparse Categorical Crossentropy (`from_logits=True`)

## ⚙️ Training Configuration

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Optimizer     | Adam                            |
| Learning Rate | 0.001                           |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs        | 20                              |
| Classes       | 10                              |

## 🌐 Frontend (User Interface)

The frontend was developed using:

* HTML5
* CSS3
* JavaScript
* HTML5 Canvas API

### How It Works

1. User draws a digit on the canvas.
2. JavaScript captures drawing coordinates.
3. Canvas content is converted into a Base64 image.
4. Image is packaged into JSON format.
5. Data is sent asynchronously using the Fetch API.
6. Prediction is displayed without reloading the page.

## 🐍 Backend (Flask Server)

The backend was developed using Flask and acts as the bridge between the browser and the TensorFlow model.

### Responsibilities

* Receive image requests
* Decode Base64 image data
* Execute preprocessing pipeline
* Run model inference
* Return predictions as JSON

### Image Preprocessing Pipeline

The uploaded image is standardized using Pillow (PIL):

#### Grayscale Conversion

Removes unnecessary color information.

#### Resize to 28×28

Matches MNIST training dimensions.

#### Pixel Thresholding

Removes anti-aliasing artifacts and improves contrast.

#### Normalization

Scales pixel values between 0 and 1.

#### Flattening

Transforms the 28×28 image into a 784-dimensional vector for Dense layers.

## 🧠 Prediction Pipeline

The trained model receives the processed input and generates raw logits.

### Softmax Probability Conversion

Because the model was trained using:

```python
SparseCategoricalCrossentropy(from_logits=True)
```

the output logits are converted using:

```python
tf.nn.softmax()
```

This generates probabilities that sum to 100%.

### Final Prediction

The digit with the highest probability is selected using:

```python
np.argmax(probabilities)
```

The application also displays the Top-3 most likely digits with confidence percentages.

## 📊 Evaluation & Visualizations

The project includes:

### Training Metrics

* Accuracy
* Loss

### Visualization Tools

* Sample image visualization
* Prediction grid (64 random images)
* Training accuracy plots
* Training loss plots

### Confusion Matrix

Provides detailed insight into classification performance and highlights common misclassifications between digits.

## 📈 Results

The model successfully classifies handwritten digits with high accuracy and confidence.

### Outputs Include

* Predicted digit
* Confidence score
* Top-3 predictions
* Visual evaluation plots

## 🛠️ Technologies Used

| Category                | Tool                            |
| ----------------------- | ------------------------------- |
| Programming Language    | Python                          |
| Deep Learning           | TensorFlow / Keras              |
| Backend                 | Flask                           |
| Frontend                | HTML5, CSS3, JavaScript         |
| Image Processing        | Pillow (PIL)                    |
| Numerical Computing     | NumPy                           |
| Visualization           | Matplotlib                      |
| Evaluation              | Scikit-learn                    |
| Development Environment | Google Colab / Jupyter Notebook |

## 🎓 Key Learning Outcomes

* Deep learning model development
* Multiclass image classification
* TensorFlow deployment workflows
* REST API development with Flask
* Client-server communication
* Image preprocessing techniques
* Frontend-backend integration
* Model evaluation and visualization

## 🔮 Future Enhancements

### 1. Convolutional Neural Networks (CNNs)

Improve accuracy by leveraging spatial image features.

### 2. Real-Time Drawing Optimization

Implement stroke smoothing and dynamic scaling.

### 3. Active Learning Feedback Loop

Collect incorrect predictions and user corrections for continual retraining.

### 4. Mobile-Friendly Interface

Optimize the application for smartphones and tablets.

### 5. Cloud Deployment

Deploy using:

* Render
* Railway
* AWS
* Google Cloud
* Azure

### 6. Explainable AI

Integrate:

* Grad-CAM
* Saliency Maps
* Feature Visualizations

## 📝 Conclusion

This project demonstrates the complete lifecycle of a machine learning application—from training a neural network on the MNIST dataset to deploying it as a real-time web application. By integrating TensorFlow, Flask, JavaScript, and image processing techniques, the system provides an interactive platform for handwritten digit recognition while showcasing practical deployment and full-stack AI development concepts.
