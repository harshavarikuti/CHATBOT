from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model



app = Flask(__name__)

# Load pickled classes and words
with open("classes_en.pkl", "rb") as f:
    classes = pickle.load(f)

with open("words_en.pkl", "rb") as f:
    words = pickle.load(f)

# Load intents and patterns from JSON
data_file = open("intents_en.json").read()
intents = json.loads(data_file)

lemmatizer = WordNetLemmatizer()

# Load trained model
model = load_model('my_model_en.h5')

def clean_up_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data.get('user_input', '')

    p = bow(user_input, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    predicted_intent_index = results[0][0]
    predicted_intent_tag = classes[predicted_intent_index]

    # Find the intent by tag and select a random response
    intent = next((intent for intent in intents["intents"] if intent["tag"] == predicted_intent_tag), None)
    if intent:
        response = random.choice(intent["responses"])
    else:
        response = "I'm not sure how to respond to that."

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
