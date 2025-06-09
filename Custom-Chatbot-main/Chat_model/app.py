from flask import Flask, request, jsonify
from flask_cors import CORS
from custom_chatbot import chatbot_bow
from flask import send_from_directory

app = Flask(__name__)
CORS(app)

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data['question']
    response = chatbot_bow(question)  
    return jsonify({'response': response})

@app.route('/dataset/<path:path>')
def serve_dataset(path):
    return send_from_directory('Dataset', path)

if __name__ == '__main__':
    app.run(debug=True)