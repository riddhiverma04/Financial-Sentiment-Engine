import os
import glob
import re
from bs4 import BeautifulSoup
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================================================================
# SUB-STEP 2A: HTML TEXT EXTRACTION & CLEANING
# ==============================================================================
def extract_clean_text_from_sec_file(file_path: str) -> str:
    """
    Parses a raw SEC EDGAR HTML/TXT filing and extracts plain text,
    stripping HTML tags, inline CSS styles, and repetitive whitespace.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    soup = BeautifulSoup(content, "lxml")
    
    # Remove script and style blocks
    for script in soup(["script", "style"]):
        script.extract()
        
    text = soup.get_text(separator=" ")
    
    # Clean whitespace and line breaks
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, max_words: int = 150) -> list:
    """
    Splits long text into small word chunks to keep within 
    BERT's maximum sequence length (512 tokens).
    """
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


# ==============================================================================
# SUB-STEP 2B: FINBERT SENTIMENT SCORING
# ==============================================================================
class FinBERTSentimentAnalyzer:
    def __init__(self):
        print("[FinBERT] Loading ProsusAI/finbert model weights and tokenizer...")
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()  # Set model to evaluation mode
        
    def score_document(self, text: str, max_chunks: int = 25) -> float:
        """
        Scores text chunks using FinBERT.
        Returns a net sentiment score bounded between -1.0 (Negative) and +1.0 (Positive).
        """
        chunks = chunk_text(text, max_words=150)[:max_chunks]
        if not chunks:
            return 0.0
            
        chunk_scores = []
        
        for chunk in chunks:
            # Tokenize input text chunk
            inputs = self.tokenizer(
                chunk, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                
            # FinBERT class map: 0 -> Positive, 1 -> Negative, 2 -> Neutral
            p_pos = probabilities[0].item()
            p_neg = probabilities[1].item()
            
            # Net chunk sentiment score: P(Positive) - P(Negative)
            net_score = p_pos - p_neg
            chunk_scores.append(net_score)
            
        # Composite document sentiment (average across chunks)
        doc_sentiment = sum(chunk_scores) / len(chunk_scores)
        return float(doc_sentiment)


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    TICKER = "NVDA"
    FILINGS_DIR = f"./data/sec-edgar-filings/{TICKER}/10-Q"
    
    print("=========================================================")
    print("      STEP 2: SEC TEXT EXTRACTION & FINBERT SENTIMENT    ")
    print("=========================================================")
    
    # Locate downloaded SEC 10-Q files from Step 1
    filing_files = glob.glob(f"{FILINGS_DIR}/*/*.txt") + glob.glob(f"{FILINGS_DIR}/*/*.html")
    
    if not filing_files:
        print(f"[Error] No filing files found in {FILINGS_DIR}. Please re-run Step 1.")
        exit()
        
    print(f"[NLP Engine] Found {len(filing_files)} 10-Q documents to process.")
    
    # Initialize FinBERT Analyzer
    analyzer = FinBERTSentimentAnalyzer()
    
    results = []
    
    for idx, file_path in enumerate(filing_files, 1):
        # Extract folder/filing identifier
        folder_name = os.path.basename(os.path.dirname(file_path))
        print(f"\n[{idx}/{len(filing_files)}] Processing filing: {folder_name}...")
        
        # 1. Clean Text
        clean_text = extract_clean_text_from_sec_file(file_path)
        print(f"  └─ Extracted {len(clean_text.split())} words.")
        
        # 2. Score Sentiment
        sentiment_score = analyzer.score_document(clean_text, max_chunks=25)
        print(f"  └─ FinBERT Net Sentiment Score: {sentiment_score:.4f}")
        
        results.append({
            "Filing_Folder": folder_name,
            "Word_Count": len(clean_text.split()),
            "Sentiment_Score": sentiment_score
        })
        
    # Save Sentiment Analysis Results
    sentiment_df = pd.DataFrame(results)
    os.makedirs("./data/processed", exist_ok=True)
    output_csv = f"./data/processed/{TICKER}_filing_sentiment.csv"
    sentiment_df.to_csv(output_csv, index=False)
    
    print("\n=========================================================")
    print(f"[NLP Engine] ✓ Filing sentiment scores cached to:\n  └─ {output_csv}")
    print("=========================================================")
    print(sentiment_df)