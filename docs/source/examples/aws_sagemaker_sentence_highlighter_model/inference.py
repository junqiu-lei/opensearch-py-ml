import json
import os
import logging
import nltk
import torch
import torch.nn.functional
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("inference.py script started")

def ensure_nltk_data():
    """Ensure NLTK punkt data is available"""
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        logger.info("Successfully downloaded NLTK data")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        raise

# Ensure NLTK data is available at startup
ensure_nltk_data()

def model_fn(model_dir):
    """
    Load the model for inference.
    
    Parameters
    ----------
    model_dir : str
        Directory containing the model files
        
    Returns
    -------
    torch.jit.ScriptModule
        The loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(os.path.join(model_dir, "model.pt"), map_location=device)
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.
    
    Parameters
    ----------
    request_body : bytes
        The request body containing the input data
    request_content_type : str
        The content type of the request
        
    Returns
    -------
    dict
        Dictionary containing the processed input tensors
        
    Raises
    ------
    ValueError
        If the input format is invalid or required fields are missing
    """
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        
        # Handle both single and batch inputs
        if "inputs" in input_data:
            # Batch input format
            if not isinstance(input_data["inputs"], list):
                raise ValueError("'inputs' must be a list of question-context pairs")
            
            # Validate each input in the batch
            for i, item in enumerate(input_data["inputs"]):
                if not isinstance(item, dict):
                    raise ValueError(f"Input {i} must be a dictionary")
                if "question" not in item:
                    raise ValueError(f"Input {i} missing 'question' field")
                if "context" not in item:
                    raise ValueError(f"Input {i} missing 'context' field")
            
            # Process each input in the batch
            processed_inputs = []
            for item in input_data["inputs"]:
                processed = process_single_input(item["question"], item["context"])
                processed_inputs.append(processed)
            
            # Stack tensors for batch processing
            return {
                "input_ids": torch.stack([p["input_ids"] for p in processed_inputs]),
                "attention_mask": torch.stack([p["attention_mask"] for p in processed_inputs]),
                "token_type_ids": torch.stack([p["token_type_ids"] for p in processed_inputs]),
                "sentence_ids": torch.stack([p["sentence_ids"] for p in processed_inputs])
            }
        else:
            # Single input format
            if "question" not in input_data:
                raise ValueError("Missing 'question' field in input")
            if "context" not in input_data:
                raise ValueError("Missing 'context' field in input")
            
            return process_single_input(input_data["question"], input_data["context"])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def process_single_input(question, context):
    """
    Process a single question-context pair into model inputs.
    
    Parameters
    ----------
    question : str
        The question text
    context : str
        The context text
        
    Returns
    -------
    dict
        Dictionary containing the processed input tensors
    """
    # Split context into sentences
    sentences = nltk.sent_tokenize(context)
    
    # Tokenize question and context
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Map word-level sentence IDs to token-level IDs
    sentence_ids = []
    current_sentence = 0
    for token in tokenizer.tokenize(context):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            sentence_ids.append(-100)
        else:
            sentence_ids.append(current_sentence)
            if token.endswith(".") or token.endswith("!") or token.endswith("?"):
                current_sentence += 1
    
    # Pad sentence_ids to match input length
    sentence_ids = [-100] + sentence_ids + [-100] * (inputs["input_ids"].size(1) - len(sentence_ids) - 1)
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs["token_type_ids"],
        "sentence_ids": torch.tensor([sentence_ids])
    }

def predict_fn(input_data, model):
    """
    Apply model to the input data.
    
    Parameters
    ----------
    input_data : dict
        Dictionary containing the input tensors
    model : torch.jit.ScriptModule
        The loaded model
        
    Returns
    -------
    dict
        Dictionary containing the prediction results
    """
    # Get model predictions
    with torch.no_grad():
        predictions = model(
            input_data["input_ids"],
            input_data["attention_mask"],
            input_data["token_type_ids"],
            input_data["sentence_ids"]
        )
    
    # Process predictions
    if isinstance(predictions, tuple):
        # Batch input case
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "highlighted_sentences": pred.tolist()
            })
        return {"results": results}
    else:
        # Single input case
        return {
            "highlighted_sentences": predictions.tolist()
        }

def output_fn(prediction_output, response_content_type):
    """
    Serialize and prepare the prediction output
    
    Output format (JSON):
    {
        "highlights": [
            {
                "start": 45,
                "end": 123,
                "text": "This sentence is relevant to the question.",
                "position": 2
            },
            ...
        ]
    }
    
    Args:
        prediction_output: List of highlighted sentences with metadata
        response_content_type: Content type for the response (should be 'application/json')
    
    Returns:
        str: JSON-formatted response
    
    Raises:
        ValueError: If content type is not supported
    """
    try:
        if response_content_type == 'application/json':
            # Format the output in the desired structure
            formatted_output = {
                "highlights": []
            }
            
            # Add each highlighted sentence to the output
            for sentence in prediction_output:
                highlight = {
                    "start": sentence['start'],
                    "end": sentence['end'],
                    "text": sentence['text'],
                    "position": sentence['position']
                }
                formatted_output["highlights"].append(highlight)
            
            response = json.dumps(formatted_output)
            return response
        raise ValueError(f"Unsupported content type: {response_content_type}")
    except Exception as e:
        logger.error(f"Error preparing output: {str(e)}", exc_info=True)
        raise
