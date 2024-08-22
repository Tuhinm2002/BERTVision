# 🚀 **Real-Time Emotion Prediction** 🎥🧠

Welcome to the **Real-Time Emotion Prediction** project! This repository merges the exciting fields of **Computer Vision** and **Natural Language Processing** to classify emotions from text in real time using a BERT model.

## 📚 **Overview**
In this project, we harness the power of **BERT** to analyze text presented in front of a camera. The system predicts the emotion conveyed by the text in real time, combining the capabilities of CV and NLP. Perfect for applications where understanding human emotion quickly is key!

## 🌟 **Features**
- **🎯 Real-Time Prediction:** Instant emotion classification from text displayed in front of a camera.
- **🧠 BERT-Based Model:** Leverages BERT's power for accurate text classification.
- **🤖 Computer Vision Integration:** Uses computer vision techniques to read and analyze text in real time.
- **⚡ High Performance:** Optimized for fast and reliable results.
- **💾 Dockerized:** Easily deployable using Docker containers.
- **🚀 Deployed on Hugging Face Spaces:** Accessible and easy to use.

## 🛠️ **Setup**
1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/real-time-emotion-prediction.git
    ```
2. **Navigate to the project directory:**
    ```bash
    cd real-time-emotion-prediction
    ```
3. **Build and run the Docker container:**
    ```bash
    docker build -t emotion-prediction .
    docker run -p 8000:8000 emotion-prediction
    ```

## 📊 **Model**
The model is based on **BERT**, fine-tuned on a **Twitter comments dataset** to effectively classify emotions into categories like `positive` and `negative`.

## 🔍 **How It Works**
1. The user presents a piece of text in front of a camera.
2. The web app captures and processes the text in real time.
3. The BERT model predicts the emotion behind the text.
4. The result is displayed instantly on the screen.

## 🌐 **Live Demo**
Check out the [live demo on Hugging Face Spaces](https://huggingface.co/spaces/your-demo-link) to see the project in action!

## 🧩 **Future Work**
- Expand the model to classify more complex emotions.
- Integrate with speech recognition for voice-based emotion detection.
- Enhance the real-time processing speed and accuracy.

## 📝 **License**
This project is licensed under the MIT License.
