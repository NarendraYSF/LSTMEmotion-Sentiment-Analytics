# Sentiment Analysis on Movie Reviews  

## Project Overview  
This project implements a sentiment analysis model using Long Short-Term Memory (LSTM) networks to classify IMDB movie reviews as positive or negative. The model leverages deep learning techniques for text analysis, providing a robust solution to evaluate user sentiments.  

## Features  
- Binary classification of movie reviews (positive or negative).  
- Preprocessing pipeline with tokenization and padding for textual data.  
- LSTM-based architecture for sequential data learning.  
- A user-friendly function for real-time sentiment predictions.  

---

## Technologies Used  
- **Programming Language:** Python  
- **Libraries:**  
  - TensorFlow/Keras  
  - Pandas  
  - Scikit-learn  
- **Tools:** Kaggle API for dataset retrieval  

---

## Dataset  
- **Source:** IMDB Dataset of 50K Movie Reviews  
- **Access:** Downloaded via Kaggle API.  
- **Structure:** Includes 50,000 movie reviews labeled as "positive" or "negative."  

---

## Workflow  

### 1. Dataset Handling  
- **Download:** Use the Kaggle API to fetch the dataset.  
- **Extraction:** Extract the CSV file from the zip archive.  

### 2. Data Preprocessing  
- Load the dataset using Pandas.  
- Convert sentiment labels into numerical values (`positive: 1`, `negative: 0`).  

### 3. Train-Test Split  
- Split the data into 80% training and 20% testing subsets.  

### 4. Tokenization and Padding  
- Tokenize text to convert words into sequences of integers.  
- Apply padding to ensure consistent sequence lengths for the LSTM model.  

### 5. Model Architecture  
The LSTM model consists of:  
- **Embedding Layer:** Converts word indices to dense vectors.  
- **LSTM Layer:** Processes sequential data for sentiment classification.  
- **Dense Output Layer:** Sigmoid activation for binary classification.  

### 6. Model Training  
- Trained for 5 epochs with a batch size of 64.  
- Validation split: 20% of the training data.  

### 7. Model Evaluation  
- Evaluate the trained model on the test dataset.  
- Metrics: Accuracy and binary cross-entropy loss.  

### 8. Sentiment Prediction  
- Implement a function to classify the sentiment of user-provided reviews.  

---

## Results  
- The LSTM model achieved significant accuracy on the test data.  
- Example Predictions:  
  - **Input:** "This movie was not so interesting." -> **Prediction:** Negative  
  - **Input:** "This movie was very amazing." -> **Prediction:** Positive  

---

## How to Run  

1. Clone this repository:  
   ```bash
   git clone https://github.com/NarendraYSF/LSTMEmotion-Sentiment-Analytics.git
   ```  

2. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

3. Run the Jupyter Notebook to train, evaluate, and test the model.  

---

## Future Scope  
- **Expand Dataset:** Include more diverse reviews to improve generalization.  
- **Optimize Hyperparameters:** Experiment with different learning rates, batch sizes, and epoch counts.  
- **Alternative Architectures:** Explore GRU, Transformer-based models (e.g., BERT).  
- **Deploy:** Build a web or API interface for real-time predictions.  

---

## License  
This project is licensed under the [GNU General Public License](https://opensource.org/licenses/GPL-3.0).  
