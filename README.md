# 🐦 Sentiment Analysis on Twitter Dataset

This project implements a comprehensive sentiment analysis pipeline on a large-scale Twitter dataset.  
It leverages advanced text preprocessing, feature engineering, and a deep learning LSTM model to classify the sentiment polarity of tweets as positive, negative, or neutral.

---

## 🚀 Features

- Extensive text preprocessing: cleaning, tokenization, stopword removal, stemming  
- Handles imbalanced data for robust model performance  
- LSTM network for effective sequential text modeling  
- Visualization of model metrics: confusion matrix, ROC curves, accuracy/loss graphs  
- Advanced evaluation: classification reports, AUC scores  
- Interactive exploratory data analysis using seaborn and matplotlib  

---

## 📂 Project Structure

Sentiment-Analysis-Twitter/
├── Sentiment_analysis.ipynb # Jupyter notebook with EDA, preprocessing, modeling, evaluation
├── training.1600000.processed.noemoticon.csv # Twitter dataset (1.6M tweets; not included)
├── requirements.txt # Python package dependencies
└── README.md # Documentation

text

---

## 🧐 Project Overview

Sentiment analysis identifies opinion polarity in text data to understand attitudes toward topics.  
This project trains an LSTM model on the Sentiment140 Twitter dataset (1.6 million labeled tweets) to classify tweets as positive, negative, or neutral.  
The model captures sequential dependencies in text to improve prediction accuracy.

---

## 🛠️ Technologies & Libraries

- Python 3.7+  
- Pandas and NumPy for data manipulation  
- NLTK for text processing (tokenization, stopwords, stemming)  
- TensorFlow and Keras for LSTM modeling  
- Matplotlib and Seaborn for plotting  
- Scikit-learn for evaluation metrics and train-test split  
- MLxtend for enhanced confusion matrix plotting  

---

## 📊 Model Architecture

- Embedding layer transforms tokens into dense word vectors  
- LSTM layer captures long-term dependencies in tweet sequences  
- Fully connected dense layers for classification  
- Output layer with softmax activation for multi-class sentiment output  
- Optimizer: RMSprop  
- Metrics monitored: accuracy, loss with validation on hold-out set  

---

## 🔧 Setup Instructions

1. Clone the repository and navigate to the folder:  
git clone https://github.com/<your-username>/Sentiment-Analysis-Twitter.git
cd Sentiment-Analysis-Twitter

text

2. Install required Python packages:  
pip install -r requirements.txt

text

3. Download the Sentiment140 dataset from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) or [alternate Sources], and place it in the project directory.

4. Run the Jupyter notebook for complete analysis and model execution:  
jupyter notebook Sentiment_analysis.ipynb

text

---

## 📈 Results & Evaluation

- Visualization of training and validation accuracy and loss  
- Confusion matrix displaying classification performance  
- ROC curves and AUC scores for model discrimination power  
- Classification report with precision, recall, and F1-score metrics  

---

## 💡 Usage

- Use trained models to classify tweet sentiments in social media analytics  
- Extend pipeline with transformer-based models for improved performance  
- Adapt preprocessing and modeling for other text classification tasks  

---

## 👤 Author

Aditya Gavhane

---

## 📄 License

This project is licensed under the MIT License.
