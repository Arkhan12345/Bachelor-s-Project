from flask import Flask, request, jsonify
from Biomistral_demo import (
    load_model,
    biomistral_chat,
    biomistral_generate_messages,
    biomistral_generate_raw,
)

app = Flask(__name__)

print("Loading BioMistral model...")
tokenizer, model, device = load_model()

@app.post("/generate")
def generate():
    data = request.get_json() or {}
    prompt = (data.get("prompt") or "").strip()
    messages = data.get("messages") or []

    raw_prompt = bool(data.get("raw_prompt", False))

    max_new_tokens = int(data.get("max_new_tokens", 512))
    temperature = float(data.get("temperature", 0.2))
    top_p = float(data.get("top_p", 0.9))

    if not prompt and not messages:
        return jsonify({"error": "Missing prompt or messages"}), 400
    if messages and not isinstance(messages, list):
        return jsonify({"error": "messages must be a list"}), 400

    try:
        if messages:
            reply = biomistral_generate_messages(
                tokenizer,
                model,
                device,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        elif raw_prompt:
            reply = biomistral_generate_raw(
                tokenizer, model, device,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            # legacy behavior
            reply = biomistral_chat(
                tokenizer, model, device,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        input_size = len(messages) if messages else len(prompt)
        print(
            f"[DEBUG] messages={bool(messages)} raw={raw_prompt} "
            f"Input size: {input_size}, Reply length: {len(reply)}"
        )
        return jsonify({"output": reply})
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting LLM server on port 8000...")
    app.run(host="0.0.0.0", port=8000, debug=False)
