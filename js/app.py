import csv
import os
import logging
import difflib
import spacy
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.error("Spacy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
    exit()

# Load chatbot data from CSV
def load_chatbot_data(filename="farming_guidance_large.csv"):
    if not os.path.exists(filename):
        logging.error(f"CSV file '{filename}' not found.")
        return pd.DataFrame()
    
    df = pd.read_csv(filename)
    if df.empty:
        logging.error("CSV file is empty or could not be read.")
    return df

df = load_chatbot_data()

# Extract questions and answers
columns_to_use = [
    "Soil Type", "Climate Requirements", "Watering Frequency",
    "Fertilizer Type", "Pest Control Methods", "Harvest Time (Days)",
    "Yield per Acre (kg)", "Market Price (USD/ton)"
]

questions = []
answers = []
for _, row in df.iterrows():
    for col in columns_to_use:
        if pd.isna(row[col]):
            continue  # Skip missing data
        question = f"What is the {col.lower()} for {row['Crop Name']}?"
        answer = f"{col} for {row['Crop Name']}: {row[col]}"
        questions.append(question)
        answers.append(answer)

# Use TF-IDF for text similarity
vectorizer = TfidfVectorizer()
if questions:
    question_vectors = vectorizer.fit_transform(questions)
else:
    logging.error("No data available for chatbot training.")

def get_best_match(user_query):
    if not questions:
        return "Sorry, the chatbot is currently unavailable."
    
    user_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vector, question_vectors).flatten()
    best_match_idx = similarities.argmax()
    
    if similarities[best_match_idx] > 0.3:  # Threshold for relevance
        return answers[best_match_idx]
    return "I'm not sure. Can you rephrase your question?"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.form.get("user_input", "").strip()
    user_id = request.remote_addr
    
    if not user_input:
        return jsonify({"response": "Please enter a question."})
    
    response = get_best_match(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)