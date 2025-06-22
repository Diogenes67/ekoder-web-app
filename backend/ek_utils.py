"""
ek_utils.py - Utility functions for EKoder
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Tuple, Optional


def load_codes(filepath: Path) -> pd.DataFrame:
    """
    Load ICD-10 ED codes from Excel file.
    
    Args:
        filepath: Path to the Excel file containing ED codes
        
    Returns:
        DataFrame with ED codes and descriptions
    """
    try:
        # Load the Excel file
        df = pd.read_excel(filepath)
        # Direct column renaming for your specific file
        if 'ED Short' in df.columns:
            df.rename(columns={
                'ED Short': 'ED Short List code',
                'Diagnosis': 'ED Short List Term',
                'Descriptor': 'ED Short List Included conditions',
                'Scale': 'Scale'
            }, inplace=True)

        # Ensure required columns exist
        required_cols = [
            'ED Short List code',
            'ED Short List Term',
            'ED Short List Included conditions',
            'Scale'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Try alternative column names
            alt_mappings = {
                'ED Short List code': ['Code', 'ICD10', 'ICD-10 Code', 'ED Short'],
                'ED Short List Term': ['Description', 'Term', 'Short Description', 'Diagnosis'],
                'ED Short List Included conditions': ['Conditions', 'Included Conditions', 'Keywords', 'Descriptor'],
                'Scale': ['Complexity', 'Level', 'Scale']
            }
            
            for req_col in missing_cols:
                for col in df.columns:
                    if col in alt_mappings.get(req_col, []):
                        df.rename(columns={col: req_col}, inplace=True)
                        break
        
        # Add description column if not present
        if 'description' not in df.columns:
            df['description'] = df.get('ED Short List Term', '')
        
        # Ensure Scale is numeric
        if 'Scale' in df.columns:
            df['Scale'] = pd.to_numeric(df['Scale'], errors='coerce').fillna(0)
        else:
            df['Scale'] = 0
            
        return df
        
    except Exception as e:
        raise Exception(f"Error loading codes file: {e}")


def parse_gpt(response: str, codes_df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """
    Parse GPT response to extract ED code recommendations.
    
    Args:
        response: GPT model response text
        codes_df: DataFrame containing valid ED codes
        
    Returns:
        List of tuples (code, term, explanation)
    """
    lines = response.strip().split('\n')
    parsed = []
    
    # Valid ED codes for validation
    valid_codes = set(codes_df['ED Short List code'].values)
    
    for line in lines:
        # Pattern 1: "1. CODE — Term (explanation)"
        match = re.match(r'^\d+\.\s*([A-Z]\d{2}(?:\.\d+)?)\s*[—-]\s*([^(]+)(?:\(([^)]+)\))?', line)
        
        if not match:
            # Pattern 2: "1. CODE - Term: explanation"
            match = re.match(r'^\d+\.\s*([A-Z]\d{2}(?:\.\d+)?)\s*[-—]\s*([^:]+)(?::\s*(.+))?', line)
        
        if match:
            code = match.group(1).strip()
            term = match.group(2).strip()
            explanation = match.group(3).strip() if match.group(3) else ""
            
            # Only include if it's a valid ED code
            if code in valid_codes:
                parsed.append((code, term, explanation))
            else:
                # Try to find similar codes (in case of typos)
                for valid_code in valid_codes:
                    if code[:3] == valid_code[:3]:  # Same category
                        parsed.append((valid_code, term, explanation))
                        break
    
    return parsed


def extract_key_symptoms(note_text: str) -> str:
    """
    Extract key symptoms and clinical findings from ED note.
    
    Args:
        note_text: Clinical note text
        
    Returns:
        String containing extracted key symptoms
    """
    patterns = [
        # Chief complaint patterns
        r'chief complaint[:\s]+([^.]+)',
        r'c/o\s+([^.]+)',
        r'complain(?:s|ing|ed)?\s+of\s+([^.]+)',
        r'present(?:s|ing|ed)?\s+with\s+([^.]+)',
        r'presents?\s+for\s+([^.]+)',
        
        # Clinical findings patterns
        r'diagnosis[:\s]+([^.]+)',
        r'impression[:\s]+([^.]+)',
        r'assessment[:\s]+([^.]+)',
        r'symptoms?\s+(?:include|are)[:\s]+([^.]+)',
        
        # Specific symptom patterns
        r'pain\s+in\s+(?:the\s+)?([^,.]+)',
        r'(\w+\s+pain)[,.]',
        r'(shortness of breath|dyspn[eo]a)',
        r'(chest pain|chest discomfort)',
        r'(abdominal pain|abd pain)',
        r'(headache|HA|cephalgia)',
    ]
    
    findings = []
    note_lower = note_text.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, note_lower, re.IGNORECASE)
        findings.extend(matches)
    
    # Clean and deduplicate findings
    cleaned_findings = []
    for finding in findings:
        finding = finding.strip().rstrip('.,;')
        if finding and len(finding) > 3 and finding not in cleaned_findings:
            cleaned_findings.append(finding)
    
    # Return top findings
    return " | ".join(cleaned_findings[:5]) if cleaned_findings else ""


def tokenize(text: str) -> set:
    """
    Simple alphanumeric tokenizer for text matching.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        Set of lowercase tokens (4+ characters)
    """
    return {w.lower() for w in re.findall(r'[A-Za-z]{4,}', text)}


def calculate_code_confidence(code: str, note_text: str, codes_df: pd.DataFrame, 
                            tfidf_scores: pd.Series) -> float:
    """
    Calculate confidence score for a code based on multiple factors.
    
    Args:
        code: ICD-10 code
        note_text: Clinical note text
        codes_df: DataFrame with code information
        tfidf_scores: TF-IDF similarity scores
        
    Returns:
        Confidence score between 0 and 1
    """
    confidence = 0.0
    
    # Get TF-IDF score for this code
    code_idx = codes_df[codes_df['ED Short List code'] == code].index
    if len(code_idx) > 0:
        tfidf_score = tfidf_scores.iloc[code_idx[0]]
        confidence += tfidf_score * 0.6  # 60% weight to TF-IDF
    
    # Check for direct keyword matches
    code_row = codes_df[codes_df['ED Short List code'] == code].iloc[0]
    code_keywords = tokenize(
        code_row.get('ED Short List Term', '') + ' ' +
        code_row.get('ED Short List Included conditions', '')
    )
    note_tokens = tokenize(note_text)
    
    keyword_overlap = len(code_keywords & note_tokens) / max(len(code_keywords), 1)
    confidence += keyword_overlap * 0.3  # 30% weight to keyword overlap
    
    # Boost for common ED presentations
    common_codes = ['R07.9', 'R06.02', 'R51', 'R10.9', 'M79.3']
    if code in common_codes:
        confidence += 0.1  # 10% boost for common codes
    
    return min(confidence, 1.0)  # Cap at 1.0


def format_code_output(code: str, term: str, explanation: str, 
                      keywords: str, scale: int = 0) -> str:
    """
    Format code output for display.
    
    Args:
        code: ICD-10 code
        term: Code description/term
        explanation: Clinical explanation
        keywords: Matching keywords
        scale: Complexity scale (0-5)
        
    Returns:
        Formatted string for display
    """
    scale_dollars = "$" * int(scale)
    
    output = f"**{code} — {term}** {scale_dollars}"
    if keywords:
        output += f" *({keywords})*"
    if explanation:
        output += f"\n{explanation}"
    
    return output


def validate_note_content(note_text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate clinical note content for processing.
    
    Args:
        note_text: Clinical note text
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not note_text or not note_text.strip():
        return False, "Note text is empty"
    
    if len(note_text.strip()) < 20:
        return False, "Note text is too short (minimum 20 characters)"
    
    # Check for potential PHI patterns (basic check)
    phi_patterns = [
        r'\b\d{3}-?\d{2}-?\d{4}\b',  # SSN
        r'\b\d{10,}\b',               # Long numbers (MRN, etc.)
        r'[A-Z][a-z]+\s+[A-Z][a-z]+\s+\d{1,2}/\d{1,2}/\d{4}',  # Name + DOB
    ]
    
    for pattern in phi_patterns:
        if re.search(pattern, note_text):
            return False, "Note may contain PHI - please de-identify before processing"
    
    return True, None

