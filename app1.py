# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import torch
import json

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = torch.load('faq_embeddings.pt')

with open('faq_data.json') as f:
    faq = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json.get("message", "")
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Cosine similarity
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    best_idx = torch.argmax(scores).item()

    response = faq['answers'][best_idx]
    link = faq['links'][best_idx]
    if link:
        response += f'<br><a href="{link}" target="_blank">Click here for more info</a>'

    return jsonify({"response": response})

@app.route('/default-message', methods=['GET'])
def default_message():
    return jsonify({"response": "Hello! How can I help you with CMLI?"})

if __name__ == '__main__':
    app.run(debug=True)
