import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

app = Flask(__name__)

label2id = {0: "class_0", 1: "class_1", 2: "class_2", 3: "class_3"}
id2label = {v: k for k, v in label2id.items()}

MODEL_PATH = "model_lora.pt"
MODEL_CHECKPOINT = "distilbert-base-uncased"

def load_model():
    try:
        model_bert = AutoModelForSequenceClassification.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels=4,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        print("Model loaded successfully!")
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            inference_mode=False,
            r=4,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"]
        )
        
        model_lora = get_peft_model(model_bert, lora_config)
        model_lora.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')), strict=False)
        model_lora.eval() 
        print("Model loaded and set to evaluation mode!")
        
        return model_lora
    except Exception as e:
        print(f"Error while loading model: {str(e)}")
        raise e


tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def predict(texts, model_lora, tokenizer):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model_lora(**encodings)
        logits = outputs.logits  # [batch_size, num_labels]
    
    predictions = torch.argmax(logits, dim=-1)
    return predictions

print("Loading model...")
model_lora = load_model()
print("Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("Request body is empty or malformed.")

        texts = data.get("texts", [])
        if not texts:
            raise ValueError("No texts provided.")
        
    
        predictions = predict(texts, model_lora, tokenizer)
        result = [id2label[pred.item()] for pred in predictions]

        return jsonify({"predictions": result})

    except Exception as e:
        print(f"Error: {str(e)}") 
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
