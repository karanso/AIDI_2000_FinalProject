# Load the necessary libraries
from flask import Flask, render_template, request
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Initialize Flask application
app = Flask(__name__)

# Load the dataset
data = pd.read_csv("data1.csv")

# Train the classifier
def train_classifier(data):
    # Function to preprocess text
    def preprocess_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        text = ' '.join(tokens)
        return text

    # Apply preprocessing to the 'Query' column
    data['Query'] = data['Query'].apply(preprocess_text)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data['Query'], data['Intent'], test_size=0.2, random_state=42)

    # Create a pipeline with CountVectorizer and Naive Bayes classifier
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # Train the classifier
    pipeline.fit(X_train, y_train)
    
    return pipeline

classifier = train_classifier(data)

# Prediction and response logic
def preprocess_input(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    return text

def predict_and_respond(query, classifier, threshold=50):
    preprocessed_query = preprocess_input(query)
    predicted_intent = classifier.predict([preprocessed_query])[0]
    confidence_score = max(classifier.predict_proba([preprocessed_query])[0]) * 100

    if confidence_score < threshold:
        return "Unknown", confidence_score, "I'm sorry, I'm not sure how to respond to that."
    
    if predicted_intent == "Finding available rooms":
        response = "Sure! We have several options available. Would you like me to provide more details?"
    elif predicted_intent == "Scheduling visits":
        response = "Certainly! Let me check the schedule and get back to you with available times."
    elif predicted_intent == "Price inquiries":
        response = "The rent varies depending on the location and amenities. Can you provide more details?"
    elif predicted_intent == "Application process":
        response = "To start the application process, please provide your contact details."
    elif predicted_intent == "Terms of the lease":
        response = "Pets are allowed in some apartments, but it depends on the landlord's policies."
    else:
        response = "I'm sorry, I'm not sure how to respond to that."

    return predicted_intent, confidence_score, response

# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting intents
@app.route('/predict', methods=['POST'])
# Route for predicting intents
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    print("Received user input:", user_input)  # Add this line to print user input
    intent, confidence, response = predict_and_respond(user_input, classifier)
    print("Sending response:", response)  # Add this line to print the response
    return {'intent': intent, 'confidence': confidence, 'response': response}


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
