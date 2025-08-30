from flask import Flask, request, jsonify
from model import chain

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "Você deve enviar uma pergunta no campo 'question'"}), 400
    
    response = chain.invoke(question)
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)