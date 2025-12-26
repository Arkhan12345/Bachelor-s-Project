from flask import Flask, render_template, request, jsonify

# import from your biomistral script file (rename as needed)
# e.g. put that code in biomistral.py and import:
from biomistral import load_model, biomistral_chat  # uses your functions :contentReference[oaicite:1]{index=1}

app = Flask(__name__)

# Load once at startup (IMPORTANT: don't load on every request)
tokenizer, model, device = load_model()

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/chat")
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Please type a question."})

    try:
        reply = biomistral_chat(tokenizer, model, device, user_message)
        return jsonify({"reply": reply})
    except Exception as e:
        # avoid leaking internal details in production; log server-side instead
        return jsonify({"reply": "Sorry — I ran into an error generating a response."}), 500

if __name__ == "__main__":
    app.run(debug=True)
