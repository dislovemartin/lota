from flask import Flask, request, jsonify
import logging
import logging.config

app = Flask(__name__)

# Configure logging
logging.config.fileConfig('../../monitoring/logging/log_config.yaml', disable_existing_loggers=False)
logger = logging.getLogger('ai_chatbot.chatbot')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    logger.info(f"Received message: {message}")
    response = generate_response(message)
    logger.info(f"Sending response: {response}")
    return jsonify({'response': response})

@app.route('/healthz', methods=['GET'])
def healthz():
    return "OK", 200

@app.route('/readyz', methods=['GET'])
def readyz():
    # Implement readiness logic here
    return "Ready", 200

def generate_response(message):
    # Placeholder for GPT-4 integration
    return f"Echo: {message}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

