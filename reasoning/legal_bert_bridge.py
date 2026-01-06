import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List

class LegalBertBridge:
    """
    Bridge for the Legal-BERT model trained for GAVL.
    Provides sequence classification for legal texts focusing on outcomes.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Default to the path we just set up
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "legal_bert")
        
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.id2label = self.model.config.id2label
        except Exception as e:
            print(f"Error loading Legal-BERT model: {e}")
            self.model = None

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a legal text and returns predicted outcome and confidence.
        """
        if self.model is None:
            return {"error": "Model not loaded"}
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get the top class
            top_prob, top_idx = torch.max(probs, dim=-1)
            label = self.id2label[top_idx.item()]
            confidence = top_prob.item()
            
            # Get all probabilities
            all_scores = {self.id2label[i]: probs[0][i].item() for i in range(len(self.id2label))}
            
        return {
            "prediction": label,
            "confidence": confidence,
            "all_scores": all_scores
        }

if __name__ == "__main__":
    # Quick test
    bridge = LegalBertBridge()
    test_text = "The petitioner claims that the lower court erred in its interpretation of the Fourth Amendment."
    result = bridge.analyze(test_text)
    print(result)
