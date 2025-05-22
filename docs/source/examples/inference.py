import os
import json
import logging
import sys
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Environment variables for model name (optional, can be hardcoded or passed via SageMaker config)
MODEL_NAME = os.environ.get("MODEL_NAME", "bert-base-uncased") # Default, replace with your actual model if known

def model_fn(model_dir):
    """
    Loads the PyTorch model from the `model_dir`.
    SageMaker will decompress the model.tar.gz into this directory.
    The model artifact (e.g., pytorch_model.bin or your .pt file) and tokenizer files
    should be in this directory.
    """
    logger.info(f"Loading model from {model_dir}...")
    try:
        # Assuming your model artifact is named 'opensearch-semantic-highlighter-v1.pt' as per the issue
        # and it's a full model state_dict or a scripted model.
        # Adjust the loading mechanism if your .pt file is different (e.g. state_dict)
        
        # Look for a .pt file first, assuming it's a torch.jit.ScriptModule or a full model save
        pt_model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
        if pt_model_files:
            model_path = os.path.join(model_dir, pt_model_files[0])
            logger.info(f"Found .pt model file: {model_path}. Attempting to load with torch.load or torch.jit.load.")
            # If it's a JIT scripted model:
            # model = torch.jit.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            # If it's a full model saved with torch.save(model, path):
            model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            logger.info("Model loaded successfully from .pt file.")
        else:
            # Fallback or alternative: Load a Hugging Face transformer model
            # This part might need to be adapted if your .pt file is not a standard Hugging Face model save
            # or if you are not using Hugging Face AutoModel classes.
            logger.info(f"No .pt file found. Attempting to load {MODEL_NAME} as a Hugging Face model from {model_dir}.")
            # If your .pt file is a state_dict, you would initialize your model class here and then load the state_dict.
            # e.g., model = MyCustomModelClass()
            # model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin")))
            model = AutoModelForTokenClassification.from_pretrained(model_dir)
            logger.info(f"Hugging Face model {MODEL_NAME} loaded from {model_dir}.")

        # Load tokenizer
        # If tokenizer files (tokenizer_config.json, vocab.txt, etc.) are in model_dir
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logger.info(f"Tokenizer loaded from {model_dir}.")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        logger.info(f"Model and tokenizer loaded successfully. Model moved to {device}.")
        return {"model": model, "tokenizer": tokenizer, "device": device}
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}", exc_info=True)
        raise

def input_fn(request_body, request_content_type):
    """
    Deserializes the input request body.
    Assumes JSON input.
    """
    logger.info(f"Received request with Content-Type: {request_content_type}")
    if request_content_type == "application/json":
        try:
            data = json.loads(request_body)
            logger.info(f"Deserialized JSON input: {data}")
            if "text" not in data:
                raise ValueError("Input JSON must have a 'text' field.")
            return data
        except Exception as e:
            logger.error(f"Error deserializing JSON input: {e}", exc_info=True)
            raise ValueError(f"Could not parse JSON input: {request_body}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}. Must be 'application/json'.")

def predict_fn(input_data, model_assets):
    """
    Makes a prediction based on the input data.
    This function should be customized based on your model's specific logic.
    For a sentence highlighter, this might involve tokenizing, getting embeddings/logits,
    and then determining which sentences are important.
    """
    logger.info(f"Received input data for prediction: {input_data}")
    model = model_assets["model"]
    tokenizer = model_assets["tokenizer"]
    device = model_assets["device"]
    
    text_input = input_data["text"]
    
    try:
        # Example: Simple tokenization and passing through the model
        # This is a placeholder. Your actual sentence highlighting logic will be more complex.
        # It might involve splitting into sentences, then processing each.
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs) # This depends on your model type (e.g., token classification, custom output)
        
        # Placeholder: Process outputs to get highlighted sentences
        # This is highly dependent on your model's architecture and task.
        # For instance, if it's token classification for highlighting tokens, you'd map tokens back to sentences.
        # If it's sentence embedding based, you might calculate sentence scores.
        
        # Example: return logits (or some processed form)
        # For a real sentence highlighter, you'd convert this to highlighted text or sentence scores.
        predictions = outputs.logits.cpu().numpy().tolist() 
        logger.info("Prediction successful.")
        
        # This is a dummy response. You'll need to structure it based on what the highlighter should return.
        # e.g., list of important sentences, or text with highlights.
        return {"highlighted_output": "Processing not fully implemented in template", "raw_predictions": predictions}

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise

def output_fn(prediction_output, response_content_type):
    """
    Serializes the prediction output to the HTTP response.
    Assumes JSON output.
    """
    logger.info(f"Serializing prediction output for Content-Type: {response_content_type}")
    if response_content_type == "application/json":
        try:
            response_body = json.dumps(prediction_output)
            logger.info("Prediction output serialized successfully.")
            return response_body
        except Exception as e:
            logger.error(f"Error serializing prediction output to JSON: {e}", exc_info=True)
            raise ValueError("Could not serialize prediction to JSON")
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}. Must be 'application/json'.")

if __name__ == '__main__':
    # Example for local testing (not used by SageMaker directly for inference)
    # You would need to place your model files in a 'model_artifacts' directory relative to this script.
    # And then simulate calls.
    
    # Create dummy model_dir for local testing
    dummy_model_dir = "temp_model_dir_for_testing"
    if not os.path.exists(dummy_model_dir):
        os.makedirs(dummy_model_dir)
        # You'd need to put your actual model files (or dummy ones) here for model_fn to work
        # e.g., open(os.path.join(dummy_model_dir, "opensearch-semantic-highlighter-v1.pt"), 'w').write("dummy model")
        # open(os.path.join(dummy_model_dir, "config.json"), 'w').write("{}") # for tokenizer
        # open(os.path.join(dummy_model_dir, "vocab.txt"), 'w').write("")   # for tokenizer

    logger.info("Starting local test example...")
    try:
        # 1. Load model (if you have model files in dummy_model_dir)
        # model_assets = model_fn(dummy_model_dir) 
        # logger.info("model_fn() test successful.")

        # 2. Prepare input
        sample_input_body = json.dumps({"text": "This is a test sentence. And another one for highlighting."})
        input_data = input_fn(sample_input_body, "application/json")
        logger.info("input_fn() test successful.")

        # 3. Predict (requires model_assets from model_fn)
        # prediction = predict_fn(input_data, model_assets)
        # logger.info(f"predict_fn() test successful: {prediction}")

        # 4. Output
        # output_json = output_fn(prediction, "application/json")
        # logger.info(f"output_fn() test successful: {output_json}")
        
        logger.info("Local testing stubs executed. Implement model loading and prediction for full test.")

    except Exception as e:
        logger.error(f"Error in local test example: {e}", exc_info=True)
    finally:
        if os.path.exists(dummy_model_dir):
            import shutil
            # shutil.rmtree(dummy_model_dir) # Clean up
            logger.info(f"Cleaned up dummy model directory: {dummy_model_dir}. (Cleanup disabled for now)")
