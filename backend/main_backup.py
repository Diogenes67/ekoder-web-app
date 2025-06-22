"""
EKoder Pro - FastAPI Backend
ICD-10 Coding Assistant for Emergency Departments
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EKoder Pro API",
    description="ICD-10 Coding Assistant for Emergency Departments",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class SearchRequest(BaseModel):
    chief_complaint: Optional[str] = Field(None, description="Chief complaint")
    age: Optional[int] = Field(None, ge=0, le=150, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    diagnosis: str = Field(..., description="Clinical diagnosis or description")

class CodeResult(BaseModel):
    code: str
    description: str
    confidence: float
    details: str
    category: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[CodeResult]
    search_id: str
    timestamp: datetime
    query: Dict[str, Any]

# ICD-10 Code Database (simplified version)
class ICD10Database:
    def __init__(self):
        self.codes = self.load_icd10_codes()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.prepare_search_index()
    
    def load_icd10_codes(self):
        """Load ICD-10 codes - in production, load from actual database"""
        # Sample ICD-10 codes for demonstration
        codes = [
            # Respiratory
            {"code": "J06.9", "description": "Acute upper respiratory infection, unspecified", 
             "keywords": "uri cold cough runny nose respiratory infection upper"},
            {"code": "J44.0", "description": "COPD with acute lower respiratory infection",
             "keywords": "copd chronic obstructive pulmonary disease exacerbation breathing"},
            {"code": "J45.901", "description": "Unspecified asthma with (acute) exacerbation",
             "keywords": "asthma attack wheezing breathing difficulty exacerbation"},
            
            # Cardiovascular
            {"code": "I21.9", "description": "Acute myocardial infarction, unspecified",
             "keywords": "heart attack mi myocardial infarction chest pain cardiac"},
            {"code": "I20.9", "description": "Angina pectoris, unspecified",
             "keywords": "chest pain angina cardiac ischemia heart"},
            {"code": "I10", "description": "Essential (primary) hypertension",
             "keywords": "high blood pressure hypertension htn elevated bp"},
            
            # Chest pain
            {"code": "R07.9", "description": "Chest pain, unspecified",
             "keywords": "chest pain nonspecific chest discomfort thoracic pain"},
            {"code": "R07.89", "description": "Other chest pain",
             "keywords": "atypical chest pain musculoskeletal chest wall pain"},
            
            # Gastrointestinal
            {"code": "K92.0", "description": "Hematemesis",
             "keywords": "vomiting blood hematemesis gi bleed upper gastrointestinal"},
            {"code": "R10.9", "description": "Unspecified abdominal pain",
             "keywords": "abdominal pain stomach pain belly pain abdomen"},
            
            # Neurological
            {"code": "G43.909", "description": "Migraine, unspecified, not intractable",
             "keywords": "migraine headache severe headache head pain"},
            {"code": "R51", "description": "Headache",
             "keywords": "headache head pain cephalgia"},
            
            # Injuries
            {"code": "S01.00XA", "description": "Unspecified open wound of scalp, initial",
             "keywords": "scalp laceration head wound cut bleeding scalp injury"},
            {"code": "S52.501A", "description": "Unspecified fracture of lower end of right radius",
             "keywords": "wrist fracture radius fracture arm broken bone"},
        ]
        
        return pd.DataFrame(codes)
    
    def prepare_search_index(self):
        """Prepare search index for quick lookups"""
        # Combine description and keywords for better search
        self.codes['search_text'] = (
            self.codes['description'] + ' ' + self.codes['keywords']
        ).str.lower()
        
        # Fit vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(self.codes['search_text'])
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for ICD-10 codes based on query"""
        # Vectorize query
        query_vector = self.vectorizer.transform([query.lower()])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append({
                    'code': self.codes.iloc[idx]['code'],
                    'description': self.codes.iloc[idx]['description'],
                    'confidence': float(similarities[idx]),
                    'details': self._generate_details(self.codes.iloc[idx])
                })
        
        return results
    
    def _generate_details(self, code_row):
        """Generate additional details for the code"""
        return f"Common keywords: {', '.join(code_row['keywords'].split()[:5])}"

# Initialize ICD-10 database
icd_db = ICD10Database()

# OpenAI Integration (optional)
class OpenAIEnhancer:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
            self.enabled = True
        else:
            self.enabled = False
            logger.warning("OpenAI API key not found. AI enhancement disabled.")
    
    async def enhance_search(self, query: str, initial_results: List[Dict]) -> List[Dict]:
        """Enhance search results using OpenAI"""
        if not self.enabled:
            return initial_results
        
        try:
            # Create prompt for OpenAI
            prompt = f"""
            Given the medical query: "{query}"
            And these potential ICD-10 codes:
            {json.dumps(initial_results, indent=2)}
            
            Please rank these codes by relevance and provide confidence scores (0-1).
            Consider the clinical context and common ED presentations.
            """
            
            # Call OpenAI API (implement actual call here)
            # response = openai.ChatCompletion.create(...)
            
            # For now, return original results
            return initial_results
        except Exception as e:
            logger.error(f"OpenAI enhancement error: {e}")
            return initial_results

ai_enhancer = OpenAIEnhancer()

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "EKoder Pro API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "search": "/api/search",
            "health": "/api/health",
            "codes": "/api/codes/{code}"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "database": {"codes_loaded": len(icd_db.codes)},
        "ai_enhancement": {"enabled": ai_enhancer.enabled}
    }

@app.post("/api/search", response_model=SearchResponse)
async def search_codes(request: SearchRequest):
    """Search for ICD-10 codes based on symptoms and diagnosis"""
    try:
        # Combine search terms
        search_query = f"{request.chief_complaint or ''} {request.diagnosis}".strip()
        
        if not search_query:
            raise HTTPException(status_code=400, detail="Please provide search terms")
        
        # Add demographic context if available
        if request.age:
            age_context = f"patient age {request.age}"
            search_query += f" {age_context}"
        
        if request.gender:
            search_query += f" {request.gender}"
        
        # Search ICD-10 database
        results = icd_db.search(search_query, top_k=10)
        
        # Enhance with AI if available
        enhanced_results = await ai_enhancer.enhance_search(search_query, results)
        
        # Convert to response format
        code_results = [
            CodeResult(
                code=r['code'],
                description=r['description'],
                confidence=r['confidence'],
                details=r['details']
            )
            for r in enhanced_results
        ]
        
        # Sort by confidence
        code_results.sort(key=lambda x: x.confidence, reverse=True)
        
        # Create response
        response = SearchResponse(
            results=code_results[:5],  # Return top 5
            search_id=f"search_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            query=request.dict()
        )
        
        logger.info(f"Search completed: {len(response.results)} results for '{search_query}'")
        return response
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/codes/{code}")
async def get_code_details(code: str):
    """Get detailed information about a specific ICD-10 code"""
    # Find code in database
    code_data = icd_db.codes[icd_db.codes['code'] == code.upper()]
    
    if code_data.empty:
        raise HTTPException(status_code=404, detail=f"Code {code} not found")
    
    row = code_data.iloc[0]
    return {
        "code": row['code'],
        "description": row['description'],
        "keywords": row['keywords'].split(),
        "category": "Emergency Medicine",  # Would be determined from code structure
        "notes": "Common ED presentation"
    }

@app.post("/api/batch-search")
async def batch_search(queries: List[str]):
    """Search multiple queries at once"""
    results = {}
    for query in queries[:10]:  # Limit to 10 queries
        try:
            search_results = icd_db.search(query, top_k=3)
            results[query] = search_results
        except Exception as e:
            results[query] = {"error": str(e)}
    
    return {"results": results}

# Export functionality
@app.post("/api/export")
async def export_results(results: List[CodeResult], format: str = "csv"):
    """Export search results in various formats"""
    if format == "csv":
        df = pd.DataFrame([r.dict() for r in results])
        csv_data = df.to_csv(index=False)
        return {"data": csv_data, "format": "csv"}
    elif format == "json":
        return {"data": [r.dict() for r in results], "format": "json"}
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
