# sentencebert_train.py
import json
import torch
from sentence_transformers import SentenceTransformer

# Load your chatbot knowledge base
with open('chatbot_knowledge.json') as f:
    data = json.load(f)

questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]
links = [item.get('link', '') for item in data]

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(questions, convert_to_tensor=True)

# Save embeddings and data
torch.save(embeddings, 'faq_embeddings.pt')

with open('faq_data.json', 'w') as f:
    json.dump({
        'questions': questions,
        'answers': answers,
        'links': links
    }, f)

print('✅ Embeddings and FAQ data saved.')
