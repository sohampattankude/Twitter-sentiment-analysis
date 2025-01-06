# Twitter-sentiment-analysis
This project analyzes the sentiment of tweets, classifying them as positive or negative using machine learning. It demonstrates natural language processing (NLP) techniques and implements a logistic regression model for sentiment classification.


## Features
1> Preprocessing of raw Twitter data (e.g., removing stopwords, stemming).

2> Sentiment classification using a logistic regression model.

3> Visualizations of sentiment distributions and model performance.

4> Modular structure for ease of understanding and customization.


## Technologies Used
1> Python: Programming language for data processing and model building.

2> NLTK: For text preprocessing (e.g., stopword removal, stemming).

3> Scikit-learn: For implementing machine learning algorithms.

4> Pandas & NumPy: For data manipulation and analysis.

5> Matplotlib & Seaborn: For data visualization

## Setup and Installation

Follow these steps to get started with the project on Google Colab:

1> Clone the Repository
   Open Google Colab and create a new notebook. Run the following command in a cell to clone the repository:
   
   !git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   %cd twitter-sentiment-analysis

## Usage
1> Data Preprocessing: Run the notebook notebooks/data_cleaning.ipynb to preprocess the dataset. This step includes:

   Removing stopwords and applying stemming using NLTK.
   
   Splitting the dataset into training and testing sets.
   
2> Model Training: Use notebooks/model_training.ipynb to train the logistic regression model on the preprocessed data.

3> Prediction: Use src/predict.py to classify new tweets. Provide the text input, and the model will predict its sentiment.

## Project Workflow
1> Data Loading: Import the dataset and inspect its structure.

2> Data Cleaning: Remove irrelevant characters, stopwords, and apply stemming.

3> Model Building: Train a logistic regression model using the cleaned and transformed data.

4> Evaluation: Evaluate model accuracy, precision, recall, and F1 score.

## Data Source

https://www.kaggle.com/datasets/kazanova/sentiment140

## Future Enhancements
1> Implementing deep learning models like LSTM or BERT for better performance.

2> Expanding the model to handle neutral sentiment classification.

3> Automating the process for real-time sentiment analysis.

## Acknowledgments
1> Thanks to NLTK and Scikit-learn for their excellent libraries.

2> Inspired by various open-source sentiment analysis projects.



