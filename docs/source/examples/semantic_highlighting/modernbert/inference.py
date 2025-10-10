import os
import json
import time
import logging
import torch
import re
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# Global model and tokenizer
model = None
tokenizer = None


def model_fn(model_dir):
    """Load ModernBERT model"""
    global model, tokenizer
    
    logger.info("Loading ModernBERT model...")
    
    # Load from model_dir (contains downloaded HuggingFace model)
    model_path = os.path.join(model_dir, "model_files") if os.path.exists(os.path.join(model_dir, "model_files")) else model_dir
    
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model.to(DEVICE)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def get_embedding(text):
    """Get sentence embedding from ModernBERT"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding


def split_sentences(text):
    """Split text into sentences with character positions"""
    # Split by periods (both English and Chinese)
    sentences = []
    current_pos = 0
    
    for match in re.finditer(r'[^.。]+[.。]?', text):
        sentence = match.group().strip()
        if sentence:
            start = text.find(sentence, current_pos)
            end = start + len(sentence)
            sentences.append({
                'text': sentence,
                'start': start,
                'end': end
            })
            current_pos = end
    
    return sentences


def highlight_sentences(question, context, min_score=0.80):
    """
    Highlight relevant sentences using ModernBERT with normalized scoring.
    
    Returns character-level positions of highlighted sentences.
    """
    # Split context into sentences
    sentences = split_sentences(context)
    
    if not sentences:
        return []
    
    # Get query embedding
    query_emb = get_embedding(question)
    
    # Batch process all sentences at once
    sentence_texts = [s['text'] for s in sentences]
    inputs = tokenizer(sentence_texts, return_tensors="pt", truncation=True, 
                      max_length=8192, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        sentence_embs = outputs.last_hidden_state.mean(dim=1)
    
    # Calculate cosine similarity
    scores = torch.nn.functional.cosine_similarity(
        query_emb.unsqueeze(0),
        sentence_embs,
        dim=-1
    )
    
    # Use raw cosine similarity scores (no normalization)
    normalized_scores = scores
    
    # Filter by threshold and get character positions
    highlights = []
    for i, score in enumerate(normalized_scores):
        if score.item() >= min_score:
            highlights.append({
                'start': sentences[i]['start'],
                'end': sentences[i]['end'],
                'score': score.item()
            })
    
    # Sort by score descending
    highlights.sort(key=lambda x: x['score'], reverse=True)
    
    # Return only start/end (remove score for API compatibility)
    return [{'start': h['start'], 'end': h['end']} for h in highlights]


def input_fn(request_body, request_content_type):
    """Parse input request"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(data, model):
    """Run inference"""
    start_time = time.time()
    
    # Check if batch or single request
    if "inputs" in data:
        # Batch inference
        inputs = data["inputs"]
        all_highlights = []
        
        for item in inputs:
            question = item.get("question", "")
            context = item.get("context", "")
            highlights = highlight_sentences(question, context)
            all_highlights.append(highlights)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "highlights": all_highlights,
            "processing_time_ms": round(processing_time, 2),
            "device": str(DEVICE)
        }
    else:
        # Single inference
        question = data.get("question", "")
        context = data.get("context", "")
        
        highlights = highlight_sentences(question, context)
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "highlights": highlights,
            "processing_time_ms": round(processing_time, 2),
            "device": str(DEVICE)
        }


def output_fn(prediction, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
