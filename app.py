import os
import io
import base64
from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

client = InferenceClient(
    provider="auto",
    api_key=os.environ.get("HF_TOKEN"),
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    model = data.get("model", "stabilityai/stable-diffusion-xl-base-1.0")

    try:
        image = client.text_to_image(prompt, model=model)

        # Convert PIL image to base64 string
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return jsonify({"image": img_base64, "prompt": prompt})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
