from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "./output/model_lora"  # Path to the LoRA model

# Load the base model and tokenizer
base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load the LoRA weights
model_lora = PeftModel.from_pretrained(base_model, MODEL_PATH)

# Set the model to evaluation mode
model_lora.eval()

# Define the prediction function
def predict(texts, model_lora, tokenizer):
    # Tokenize the input texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    # Get the model's outputs
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model_lora(**encodings)
        logits = outputs.logits  # [batch_size, num_labels]
    
    # Calculate the predicted labels
    predictions = torch.argmax(logits, dim=-1)
    
    # Return the predictions
    return predictions

# Define Flask route
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Check if the request body is empty or malformed
        if not data:
            raise ValueError("Request body is empty or malformed.")
        
        # Get the list of texts
        texts = data.get("texts", [])
        if not texts:
            raise ValueError("No texts provided.")
        
        # Perform prediction
        predictions = predict(texts, model_lora, tokenizer)
        
        # Define label mapping
        id2label = {0: "irrelevant", 1: "negative", 2: "neutral", 3: "positive"}  # Adjust based on your model
        result = [id2label[pred.item()] for pred in predictions]

        # Return the prediction results
        return jsonify({"predictions": result})
    
    except Exception as e:
        # Error handling
        print(f"Error: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# Flask app entry point
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
