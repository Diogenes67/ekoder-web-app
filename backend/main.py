"""
EKoder-4o Enhanced FastAPI Backend
Full-featured ICD-10 ED Principal Diagnosis Coder
Matches the Streamlit ekoder_4o_enhanced.py implementation
"""

import json
import re
import io
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import base64

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import openai
import logging
import os
from dotenv import load_dotenv
from ek_utils import load_codes


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EKoder-4o API",
    description="ED Principal Diagnosis Coder with TF-IDF and GPT-4",
    version="4.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import utilities
try:
    from ek_utils import load_codes, parse_gpt, extract_key_symptoms, tokenize
except ImportError:
    logger.warning("ek_utils not found, using embedded functions")
    # Embedded utility functions
    def tokenize(txt: str) -> set:
        """Simple alphanumeric tokeniser ≥4 chars."""
        return {w.lower() for w in re.findall(r"[A-Za-z]{4,}", txt)}
    
    def extract_key_symptoms(note_text: str) -> str:
        """Extract key symptoms from clinical note."""
        patterns = [
            r'complain(?:s|ing|ed)?\s+of\s+([^.]+)',
            r'present(?:s|ing|ed)?\s+with\s+([^.]+)',
            r'chief complaint[:\s]+([^.]+)',
            r'c/o\s+([^.]+)',
            r'diagnosis[:\s]+([^.]+)',
            r'impression[:\s]+([^.]+)',
            r'assessment[:\s]+([^.]+)',
        ]
        hits = []
        for p in patterns:
            hits.extend(re.findall(p, note_text, re.IGNORECASE))
        return " ".join(hits[:3]) if hits else ""
    
    def parse_gpt(response: str, codes_df: pd.DataFrame) -> List[tuple]:
        """Parse GPT response to extract code recommendations."""
        lines = response.strip().split('\n')
        parsed = []
        
        for line in lines:
            # Look for pattern like "1. CODE — Term (explanation)"
            match = re.match(r'^\d+\.\s*([A-Z]\d{2}(?:\.\d+)?)\s*[—-]\s*([^(]+)(?:\(([^)]+)\))?', line)
            if match:
                code = match.group(1).strip()
                term = match.group(2).strip()
                explanation = match.group(3).strip() if match.group(3) else ""
                
                # Only include if code exists in our database
                if code in codes_df["ED Short List code"].values:
                    parsed.append((code, term, explanation))
        
        return parsed

# Data Models
class SearchRequest(BaseModel):
    note_text: str = Field(..., description="Clinical note text")
    model: str = Field(default="gpt-4o", description="GPT model to use")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="GPT temperature")
    debug_mode: bool = Field(default=False, description="Enable debug mode")

class CodeResult(BaseModel):
    rank: int
    code: str
    term: str
    explanation: str
    keywords: str
    score: float
    scale: int

class SearchResponse(BaseModel):
    validated_results: List[CodeResult]
    shortlist: List[Dict[str, Any]]
    gpt_raw: str
    key_symptoms: str
    debug_info: Optional[Dict[str, Any]] = None

class BatchFileRequest(BaseModel):
    filename: str
    content: str  # Base64 encoded

# Global variables
codes_df = None
vectorizer = None
tfidf_matrix = None
desc_lookup = {}
tfidf_salt = 0
examples_cache = ""

# File paths
DATA_DIR = Path(__file__).resolve().parent / "data"
CODES_XLSX = DATA_DIR / "FinalEDCodes_Complexity.xlsx"
EXAMPLES_JSONL = DATA_DIR / "edcode_finetune_v7_more_notes.jsonl"

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    logger.warning("OpenAI API key not found. GPT features will be limited.")

def load_examples(path: Path, n: int = 3) -> str:
    """Load few-shot examples from JSONL file."""
    if not path.exists():
        logger.warning(f"Examples file not found: {path}")
        # Return default examples
        return """Casenote:
65yo male presents with acute onset central crushing chest pain radiating to left arm, associated with diaphoresis and nausea. ECG shows ST elevation.

Answer:
1. I21.9 — Acute myocardial infarction, unspecified

---

Casenote:
28yo female with sudden onset severe headache, photophobia, and neck stiffness. Temp 38.5C.

Answer:
1. G03.9 — Meningitis, unspecified

---

"""
    
    blocks = []
    try:
        with path.open() as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                d = json.loads(line)
                blocks.append(
                    f"Casenote:\n{d['messages'][0]['content']}\n"
                    f"Answer:\n{d['messages'][1]['content']}"
                )
    except Exception as e:
        logger.error(f"Error loading examples: {e}")
    
    return "\n\n---\n\n".join(blocks) + "\n\n---\n\n" if blocks else ""

def load_codes_df():
    """Load ICD-10 ED codes from Excel file or create sample data."""
    global codes_df, desc_lookup

try:
    codes_df = load_codes(CODES_XLSX)
    logger.info(f"Loaded {len(codes_df)} codes via load_codes from {CODES_XLSX}")
except Exception as e:
    logger.error(f"Loading real codes failed, falling back to sample: {e}")
    codes_df = create_sample_codes()




    # Ensure description always exists
    if 'description' not in codes_df.columns:
        codes_df['description'] = codes_df.get('ED Short List Term', '')

    # Build description lookup
    desc_lookup = dict(
        zip(
            codes_df["ED Short List code"],
            codes_df["ED Short List Included conditions"].fillna("")
        )
    )


def create_sample_codes():
    """Create sample ED codes for demo."""
    data = [
        {"ED Short List code": "R07.9", "ED Short List Term": "Chest pain, unspecified", 
         "ED Short List Included conditions": "chest pain, chest discomfort, thoracic pain, non-cardiac chest pain", 
         "Scale": 2, "description": "Chest pain, unspecified"},
        {"ED Short List code": "R06.02", "ED Short List Term": "Shortness of breath", 
         "ED Short List Included conditions": "dyspnea, breathing difficulty, respiratory distress, SOB", 
         "Scale": 3, "description": "Shortness of breath"},
        {"ED Short List code": "I21.9", "ED Short List Term": "Acute myocardial infarction, unspecified",
         "ED Short List Included conditions": "heart attack, MI, myocardial infarction, STEMI, NSTEMI, cardiac event", 
         "Scale": 5, "description": "Acute myocardial infarction, unspecified"},
        {"ED Short List code": "J44.0", "ED Short List Term": "COPD with acute lower respiratory infection",
         "ED Short List Included conditions": "COPD exacerbation, chronic obstructive pulmonary disease, emphysema exacerbation", 
         "Scale": 4, "description": "COPD with acute lower respiratory infection"},
        {"ED Short List code": "R51.9", "ED Short List Term": "Headache, unspecified",
         "ED Short List Included conditions": "headache, cephalgia, head pain, HA", 
         "Scale": 2, "description": "Headache, unspecified"},
        {"ED Short List code": "R10.9", "ED Short List Term": "Unspecified abdominal pain",
         "ED Short List Included conditions": "abdominal pain, stomach pain, belly pain, abd pain", 
         "Scale": 2, "description": "Unspecified abdominal pain"},
        {"ED Short List code": "M79.3", "ED Short List Term": "Myalgia",
         "ED Short List Included conditions": "muscle pain, myalgia, body aches, muscle aches", 
         "Scale": 1, "description": "Myalgia"},
        {"ED Short List code": "N39.0", "ED Short List Term": "Urinary tract infection",
         "ED Short List Included conditions": "UTI, cystitis, bladder infection, dysuria", 
         "Scale": 3, "description": "Urinary tract infection"},
        {"ED Short List code": "J18.9", "ED Short List Term": "Pneumonia, unspecified organism",
         "ED Short List Included conditions": "pneumonia, lung infection, chest infection, CAP", 
         "Scale": 4, "description": "Pneumonia, unspecified organism"},
        {"ED Short List code": "G43.909", "ED Short List Term": "Migraine, unspecified",
         "ED Short List Included conditions": "migraine, severe headache, migraine headache", 
         "Scale": 3, "description": "Migraine, unspecified"},
    ]
    return pd.DataFrame(data)

def build_tfidf(df: pd.DataFrame, salt: int):
    """Build TF-IDF vectorizer and matrix with salt for cache busting."""
    global vectorizer, tfidf_matrix
    
    df = df.copy()
    df["combined_text"] = (
        df["ED Short List Term"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["ED Short List Included conditions"].fillna("")
    )
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),   # 1-grams, 2-grams, 3-grams
        sublinear_tf=True,    # dampen high TF
        min_df=2              # drop singletons
    )
    
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    logger.info(f"Built TF-IDF with {len(vectorizer.get_feature_names_out())} features (salt: {salt})")

def process_note(note_text: str, model: str = "gpt-4o", temperature: float = 0.0, debug: bool = False):
    """Process a clinical note to find ED diagnosis codes."""
    if not note_text.strip():
        raise ValueError("Note text is empty")
    
    # Extract key symptoms
    key_symptoms = extract_key_symptoms(note_text)
    
    # TF-IDF scoring
    note_vec = vectorizer.transform([note_text.lower()])
    codes_df_work = codes_df.copy()
    codes_df_work["Score"] = cosine_similarity(note_vec, tfidf_matrix).flatten()
    
    # Debug info
    debug_info = {}
    if debug:
        debug_info["max_score"] = float(codes_df_work["Score"].max())
        debug_info["top_10_avg"] = float(codes_df_work["Score"].nlargest(10).mean())
        debug_info["tfidf_features"] = len(vectorizer.get_feature_names_out())
    
    # Get top 100 codes for GPT consideration
    shortlist = codes_df_work.sort_values("Score", ascending=False).head(100)[
        ["ED Short List code", "ED Short List Term", "Score"]
    ]
    
    opts_text = "\n".join(
        f"{row['ED Short List code']} — {row['ED Short List Term']}"
        for _, row in shortlist.iterrows()
    )
    
    # Build GPT prompt
    prompt = f"""{examples_cache}You are an expert Australian emergency physician and senior clinical coder with 20+ years of experience.

CRITICAL INSTRUCTIONS:
1. Your response MUST start with "1. "
2. NO bullets, NO asterisks, NO other formatting at the start
3. Choose the SINGLE BEST ED principal diagnosis from the provided shortlist
4. Provide differentials (lines 2–4) only if genuinely uncertain

KEY SYMPTOMS IDENTIFIED: {key_symptoms if key_symptoms else "None clearly identified"}

AVAILABLE ED CODES (top 100 by relevance):
{opts_text}

Casenote:
{note_text}

Remember: Start your response with "1. " followed by the MOST SPECIFIC appropriate code from the list above."""

    # Call GPT if API key is available
    gpt_raw = ""
    if openai.api_key:
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", 
                     "content": "You are an expert emergency medicine coder. Always follow the exact format requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=500
            )
            gpt_raw = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"GPT error: {e}")
            # Fallback to top TF-IDF match
            top_code = shortlist.iloc[0]
            gpt_raw = f"1. {top_code['ED Short List code']} — {top_code['ED Short List Term']} (TF-IDF fallback)"
    else:
        # No API key - use TF-IDF only
        top_code = shortlist.iloc[0]
        gpt_raw = f"1. {top_code['ED Short List code']} — {top_code['ED Short List Term']} (TF-IDF only - no GPT)"
    
    # Parse GPT response
    try:
        parsed = parse_gpt(gpt_raw, codes_df)
    except Exception:
        parsed = []
    
    # Validate and enhance results
    note_tokens = tokenize(note_text)
    validated = []
    
    for i, item in enumerate(parsed):
        if len(item) < 3:
            continue
        
        code, term, expl = item[:3]
        
        # Find matching keywords
        code_tokens = tokenize(term) | tokenize(desc_lookup.get(code, ""))
        hits = code_tokens & note_tokens
        kw = ", ".join(sorted(hits)) if hits else "code match"
        
        # Get scale value
        scale_val = 0
        scale_match = codes_df.loc[codes_df["ED Short List code"] == code, "Scale"]
        if not scale_match.empty:
            try:
                scale_val = int(scale_match.iloc[0])
            except Exception:
                scale_val = 0
        
        # Get TF-IDF score
        score_match = codes_df_work.loc[codes_df_work["ED Short List code"] == code, "Score"]
        score = float(score_match.iloc[0]) if not score_match.empty else 0.0
        
        validated.append({
            "rank": i + 1,
            "code": code,
            "term": term,
            "explanation": expl,
            "keywords": kw,
            "score": score,
            "scale": scale_val
        })
    
    # Convert shortlist for response
    shortlist_data = shortlist.to_dict('records')
    for item in shortlist_data:
        item["Score"] = float(item["Score"])
    
    return {
        "validated": validated,
        "shortlist": shortlist_data[:20],  # Top 20 for display
        "gpt_raw": gpt_raw,
        "key_symptoms": key_symptoms,
        "debug_info": debug_info if debug else None
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    global examples_cache
    
    # Load codes
    load_codes_df()
    
    # Build TF-IDF
    build_tfidf(codes_df, tfidf_salt)
    
    # Load examples
    examples_cache = load_examples(EXAMPLES_JSONL, 3)
    
    logger.info("Startup complete")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "EKoder-4o API",
        "version": "4.0.0",
        "status": "active",
        "codes_loaded": len(codes_df) if codes_df is not None else 0,
        "openai_configured": bool(openai.api_key)
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "codes_loaded": len(codes_df) if codes_df is not None else 0,
        "tfidf_features": len(vectorizer.get_feature_names_out()) if vectorizer else 0,
        "tfidf_salt": tfidf_salt,
        "openai_configured": bool(openai.api_key)
    }

@app.post("/api/search", response_model=SearchResponse)
async def search_codes(request: SearchRequest):
    """Search for ED diagnosis codes based on clinical note."""
    try:
        result = process_note(
            request.note_text,
            request.model,
            request.temperature,
            request.debug_mode
        )
        
        return SearchResponse(
            validated_results=[CodeResult(**v) for v in result["validated"]],
            shortlist=result["shortlist"],
            gpt_raw=result["gpt_raw"],
            key_symptoms=result["key_symptoms"],
            debug_info=result.get("debug_info")
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-file")
async def process_file(file: UploadFile = File(...)):
    """Process uploaded text file."""
    try:
        content = await file.read()
        text = content.decode('utf-8', errors='ignore')
        
        result = process_note(text)
        
        return {
            "filename": file.filename,
            "validated_results": result["validated"],
            "key_symptoms": result["key_symptoms"],
            "gpt_raw": result["gpt_raw"]
        }
        
    except Exception as e:
        logger.error(f"File processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-process")
async def batch_process(files: List[BatchFileRequest]):
    """Process multiple files in batch."""
    results = []
    
    for file_req in files:
        try:
            # Decode base64 content
            content = base64.b64decode(file_req.content)
            text = content.decode('utf-8', errors='ignore')
            
            # Process the note
            result = process_note(text)
            
            # Build row for Excel
        row = {"File": file_req.filename}
        for i, v in enumerate(result["validated"][:5], 1):
            # unpack (code, term, explanation, keyword_hits)
            code, term, explanation, _ = v

            row[f"Code {i}"]   = code
            row[f"Reason {i}"] = explanation

            # look up the Scale value
            scale_match = codes_df.loc[
                codes_df["ED Short List code"] == code, "Scale"
            ]
            row[f"Scale {i}"]  = int(scale_match.iloc[0]) if not scale_match.empty else 0

        # now append the completed row
        results.append(row)
            
        except Exception as e:
            logger.error(f"Batch error for {file_req.filename}: {e}")
            results.append({"File": file_req.filename, "Error": str(e)})
    
    # Create Excel file
    df = pd.DataFrame(results)
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='ED Codes')
    
    buffer.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename=ekoder_batch_{timestamp}.xlsx"
        }
    )

@app.post("/api/rebuild-tfidf")
async def rebuild_tfidf():
    """Rebuild TF-IDF matrix."""
    global tfidf_salt
    
    try:
        tfidf_salt += 1
        build_tfidf(codes_df, tfidf_salt)
        
        return {
            "status": "success",
            "salt": tfidf_salt,
            "features": len(vectorizer.get_feature_names_out())
        }
    except Exception as e:
        logger.error(f"TF-IDF rebuild error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/shortlist/{limit}")
async def get_shortlist(limit: int = 100):
    """Get top N codes by TF-IDF similarity."""
    if codes_df is None:
        raise HTTPException(status_code=500, detail="Codes not loaded")
    
    return {
        "codes": codes_df.head(limit)[["ED Short List code", "ED Short List Term"]].to_dict('records'),
        "total": len(codes_df)
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
