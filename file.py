import io
import torch
import flask
from PIL import Image
from flask import Flask, request, jsonify, render_template
from transformers import CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2Tokenizer
app = Flask(__name__)
# Load models (cache these for production)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Process image
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')      
        # Generate description with CLIP
        inputs = clip_processor(
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(device)
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        
        # Generate caption with GPT-2
        prompt = "This is an image of"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = gpt_model.generate(
                input_ids,
                max_length=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        caption = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return jsonify({
            "caption": caption,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)