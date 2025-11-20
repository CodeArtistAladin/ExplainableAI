# ExplainableAI
This project uses a ResNet18 image classifier with LIME to provide explainable AI insights. The Streamlit app lets users upload images, view predictions, and see highlighted regions that influenced the model’s decision. Built with Python, PyTorch, LIME, and Streamlit to make deep learning more transparent.

🧠 Explainable AI Image Classifier

A simple and interactive Explainable AI (XAI) demo using ResNet18, LIME, and Streamlit.
This project classifies images and highlights the regions that influenced the model’s decision, making deep learning more transparent and interpretable.

🚀 Features

🖼️ Image Upload & Classification
Upload an image and get predictions using a trained ResNet18 model.

🔍 LIME Explanation
Generates superpixel-based explanations showing why the model predicted a class.

⚡ Fast & Lightweight UI
Built with Streamlit for easy local deployment and demonstration.

📦 Modular Codebase
Separate modules for inference, explanation, and app UI.

🛠️ Technologies Used

Python 3.10

PyTorch – model loading & inference

Torchvision – preprocessing & transforms

LIME – explainability

Streamlit – UI

Pillow / NumPy / Matplotlib – image handling & visualization

📁 Project Structure
explainable-ai-demo/
│
├── src/
│   ├── app.py            # Streamlit UI
│   ├── predict.py        # Model prediction logic
│   ├── explain.py        # LIME explanation code
│   ├── model.pth         # Trained ResNet18 model
│   ├── classes.txt       # Class labels
│   ├── utils.py          # Helper functions
│
└── README.md

▶️ How to Run the App

Create & activate virtual environment

python -m venv .venv
.venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Run the app

streamlit run app.py


Open the browser at
http://localhost:8501

📘 How It Works

The uploaded image is preprocessed and passed into a ResNet18 classifier.

The predicted class is displayed with confidence.

LIME generates an interpretable explanation by:

Segmenting the image into superpixels

Testing how each region affects prediction

Highlighting the regions most responsible

This improves transparency for deep learning systems.

🎯 Purpose

This project demonstrates how explainable AI can increase trust and interpretability in computer vision models. Ideal for academic presentations, demos, or learning XAI concepts.

Author: 
MD MONEM SHAHREER SURJO
