from flask import Flask, request, jsonify
from Biomistral_demo import load_model, biomistral_chat

app = Flask(__name__)

print("Loading BioMistral model...")
tokenizer, model, device = load_model()

@app.post("/generate")
def generate():
    """API endpoint for LLM generation."""
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    max_new_tokens = data.get("max_new_tokens", 200)
    
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400
    
    try:
        reply = biomistral_chat(tokenizer, model, device, prompt)
        print(f"[DEBUG] Prompt length: {len(prompt)}, Reply length: {len(reply)}, Reply: '{reply[:100]}'")
        return jsonify({"output": reply})
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting LLM server on port 8000...")
    app.run(host="0.0.0.0", port=8000, debug=False)
