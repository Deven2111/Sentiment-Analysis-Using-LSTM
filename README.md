# 🎬 IMDB Movie Review Sentiment Analysis using LSTM

## 📌 Overview
This project is an **NLP-based Sentiment Analysis system** built using **Deep Learning (LSTM)**.  
The goal is to classify IMDB movie reviews as **positive** or **negative** based on their text content.  

It uses the **IMDB Dataset of 50,000 Movie Reviews** from Kaggle and implements a **Bidirectional LSTM neural network** trained with TensorFlow/Keras.  
The model achieves strong accuracy and includes a **real-time predictive system** for unseen reviews.  

---

## 🚀 Features
- 📂 **Dataset Integration** – IMDB dataset from Kaggle (50k reviews).  
- 📝 **Text Preprocessing** – Tokenization, padding, and embeddings.  
- 🧠 **Deep Learning Model** – Bidirectional LSTM with dropout for regularization.  
- 📊 **Training & Evaluation** – Accuracy/loss tracking with validation split.  
- 🔮 **Prediction System** – Input custom text reviews to get sentiment results.  

---

## 🛠️ Tech Stack
- **Programming Language**: Python  
- **Frameworks & Libraries**: TensorFlow, Keras, Scikit-learn, Pandas, NumPy  
- **NLP Tools**: Tokenizer, Embedding Layer, Sequence Padding  
- **Environment**: Google Colab / Jupyter Notebook  
- **Dataset Source**: [IMDB Dataset of 50K Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  

---

## 📂 Dataset
The dataset contains **50,000 reviews** (25k positive, 25k negative).  
- **File**: `IMDB Dataset.csv`  
- **Columns**:  
  - `review` → Text of the movie review  
  - `sentiment` → Label (*positive* / *negative*)  

---

## ⚙️ Project Workflow
1. **Data Collection** – Download IMDB dataset from Kaggle.  
2. **Preprocessing** – Tokenize reviews, convert to sequences, and pad them.  
3. **Model Building** –  
   - Embedding Layer (word vector representation)  
   - Bidirectional LSTM (captures context both forward & backward)  
   - Dense Output Layer (sigmoid activation for binary classification)  
4. **Model Training** – Optimizer: Adam | Loss: Binary Crossentropy | Metrics: Accuracy.  
5. **Evaluation** – Test accuracy measured on unseen data.  
6. **Prediction System** – Function to classify new reviews as *positive* or *negative*.  

---

## ▶️ Usage

### Clone the Repository
```bash
git clone https://github.com/Deven2111/Sentiment-Analysis-Using-LSTM.git
cd imdb-sentiment-lstm
