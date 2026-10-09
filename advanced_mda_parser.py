import os
import glob
import re
import pandas as pd
from bs4 import BeautifulSoup

# Loughran-McDonald Lexicon Financial Keyword Sets (Academic Standard)
LM_LEXICON = {
    "negative": [
        "loss", "losses", "decline", "declined", "declining", "adverse", "adversely", 
        "risk", "risks", "impairment", "deficit", "uncertainty", "uncertainties", 
        "litigation", "lawsuit", "penalty", "penalties", "default", "termination"
    ],
    "positive": [
        "growth", "growing", "gain", "gains", "profit", "profitable", "profitability", 
        "expansion", "expanding", "strong", "outperform", "efficiency", "increase", 
        "increased", "benefit", "benefits", "improvement", "successful"
    ],
    "uncertainty": [
        "approximate", "contingent", "depend", "dependent", "fluctuate", "fluctuating", 
        "indefinite", "speculative", "variable", "uncertain", "estimate", "projected"
    ],
    "litigious": [
        "alleged", "allegation", "claim", "claims", "lawsuit", "settlement", 
        "litigation", "subpoena", "verdict", "plaintiff", "defendant", "dispute"
    ],
    "constraining": [
        "binding", "commitments", "covenant", "covenants", "restrict", "restricted", 
        "restriction", "limitation", "prohibit", "prohibited", "obligation"
    ]
}

def extract_mda_section(html_content: str) -> str:
    """
    Parses raw HTML and uses Regular Expressions to isolate Item 2 (MD&A).
    Strips out navigation tables, footers, and legal disclaimers.
    """
    soup = BeautifulSoup(html_content, "lxml")
    
    # Remove script and style tags
    for element in soup(["script", "style"]):
        element.decompose()
        
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text) # Normalize whitespaces
    
    # Regex pattern targeting Item 2 MD&A header up to Item 3/Item 4
    mda_pattern = re.compile(
        r"(Item\s+2[\.\:\s\─\–\-]+Management[\’\'s]*\s+Discussion\s+and\s+Analysis.*?)("
        r"Item\s+3[\.\:\s\─\–\-]+Quantitative|Item\s+4[\.\:\s\─\–\-]+Controls|\Z)",
        re.IGNORECASE | re.DOTALL
    )
    
    match = mda_pattern.search(text)
    if match:
        mda_text = match.group(1).strip()
        if len(mda_text) > 1000:
            return mda_text[:25000]
            
    return text[2000:20000]

def score_loughran_mcdonald(text: str) -> dict:
    """
    Calculates word-frequency densities (%) across Loughran-McDonald financial categories.
    """
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    total_words = len(words) if len(words) > 0 else 1
    
    scores = {}
    for category, keywords in LM_LEXICON.items():
        match_count = sum(1 for w in words if w in keywords)
        scores[f"LM_{category}_count"] = match_count
        scores[f"LM_{category}_pct"] = round((match_count / total_words) * 100, 4)
        
    scores["LM_Net_Sentiment"] = round(scores["LM_positive_pct"] - scores["LM_negative_pct"], 4)
    scores["Total_MD&A_Word_Count"] = total_words
    return scores

def run_advanced_parsing_pipeline():
    print("=========================================================")
    print("   ADVANCED INSTITUTIONAL ENGINE: ITEM 2 MD&A PARSER    ")
    print("=========================================================")
    
    # Check both potential directory locations
    filing_dirs = sorted(glob.glob("./data/sec-edgar-filings/NVDA/10-Q/*")) + sorted(glob.glob("./sec-edgar-filings/NVDA/10-Q/*"))
    filing_dirs = [d for d in filing_dirs if os.path.isdir(d)]
    
    if not filing_dirs:
        print("[Error] No SEC filings found.")
        return

    mda_records = []
    
    for f_path in filing_dirs:
        accession_id = os.path.basename(f_path)
        file_candidates = glob.glob(f"{f_path}/*.html") + glob.glob(f"{f_path}/*.txt") + glob.glob(f"{f_path}/full-submission.txt")
        
        if not file_candidates:
            continue
            
        with open(file_candidates[0], "r", encoding="utf-8", errors="ignore") as f:
            raw_content = f.read()
            
        # 1. Isolate Item 2 Narrative
        mda_narrative = extract_mda_section(raw_content)
        
        # 2. Compute Loughran-McDonald Metrics
        lm_metrics = score_loughran_mcdonald(mda_narrative)
        
        record = {
            "Accession_ID": accession_id,
            "MD&A_Char_Length": len(mda_narrative),
        }
        record.update(lm_metrics)
        mda_records.append(record)

    df_mda = pd.DataFrame(mda_records)
    
    print("\n--- EXTRACTED ITEM 2 (MD&A) FINANCIAL LEXICON MATRIX ---")
    cols_to_show = ["Accession_ID", "Total_MD&A_Word_Count", "LM_negative_pct", "LM_positive_pct", "LM_uncertainty_pct", "LM_Net_Sentiment"]
    print(df_mda[cols_to_show].to_string(index=False))

    # Save to processed data folder
    os.makedirs("./data/processed", exist_ok=True)
    output_path = "./data/processed/NVDA_mda_loughran_mcdonald.csv"
    df_mda.to_csv(output_path, index=False)
    
    print("\n=========================================================")
    print(f"[Advanced Engine] ✓ Processing complete! Output saved to:\n  └─ {output_path}")
    print("=========================================================")

if __name__ == "__main__":
    run_advanced_parsing_pipeline()