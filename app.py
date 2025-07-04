from flask import Flask, render_template, request, jsonify, session
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
import torch
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default-secret-key')

# Initialize model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

@app.route('/')
def home():
    # Initialize or clear chat history to ensure clean start
    session['chat_history'] = []
    return render_template('index.html', chat_history=[])

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('user_input', '').strip()
    
    if not user_input:
        return jsonify({'error': 'Empty message'}), 400
    
    try:
        # Always start with fresh conversation context
        input_text = f"User: {user_input}"
        
        # Tokenize the input
        inputs = tokenizer([input_text], return_tensors='pt', truncation=True, max_length=512)
        
        # Generate response
        reply_ids = model.generate(
            **inputs,
            max_length=200,
            min_length=20,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
            num_beams=5,
            early_stopping=True
        )
        
        # Decode the response properly
        response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        
        # Clean the response
        response = response.replace("Bot:", "").strip()
        
        # Store only the text messages
        session['chat_history'] = [user_input, response]
        session.modified = True
        
        return jsonify({
            'response': response,
            'history': session['chat_history']
        })
    
    except Exception as e:
        app.logger.error(f"Error in chat: {str(e)}")
        return jsonify({'error': 'Sorry, I encountered an error processing your message'}), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    session['chat_history'] = []
    session.modified = True
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002) 