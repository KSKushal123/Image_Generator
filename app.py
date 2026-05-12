import os
import io
import base64
from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient

# load_dotenv only in local dev (not needed on Vercel)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)

def get_client():
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        raise ValueError("HF_TOKEN environment variable is not set.")
    return InferenceClient(provider="auto", api_key=api_key)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    model = data.get("model", "stabilityai/stable-diffusion-xl-base-1.0")

    try:
        client = get_client()
        image = client.text_to_image(prompt, model=model)

        # Convert PIL image to base64 string
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return jsonify({"image": img_base64, "prompt": prompt})

    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
