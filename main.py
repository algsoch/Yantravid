import os
import sys
import logging
import zipfile
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import tempfile
import json

# Configure logging for Vercel deployment
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout  # Log to stdout instead of a file for serverless environments
)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="IIT Madras Assignment Helper",
    description="API that helps answer IIT Madras Data Science graded assignment questions",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API setup - use environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables")
    # Don't raise an error here - simply log it and let the endpoints fail gracefully

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Global variable for the model
gemini_model = None

# Function to get or initialize the model
def get_model():
    global gemini_model
    if gemini_model is not None:
        return gemini_model
    
    # Try different model names
    model_names = [
        "models/gemini-1.5-flash",
        "gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "gemini-1.5-pro"
    ]
    
    for name in model_names:
        try:
            model = genai.GenerativeModel(name)
            # Test the model
            test = model.generate_content("Test")
            if test and hasattr(test, "text"):
                logging.info(f"Successfully initialized model: {name}")
                gemini_model = model
                return model
        except Exception as e:
            logging.warning(f"Failed to initialize model {name}: {str(e)}")
    
    logging.error("All model initialization attempts failed")
    return None

# Health check route
@app.get("/")
async def health_check():
    """Health check endpoint to verify the service is running"""
    return {
        "status": "ok", 
        "service": "IIT Madras Assignment Helper API"
    }

# Testing route
@app.get("/test")
async def test():
    """Test endpoint to verify the AI model is working"""
    try:
        model = get_model()
        if not model:
            return {"error": "Could not initialize AI model"}
            
        response = model.generate_content("What is 2+2?")
        return {
            "question": "What is 2+2?",
            "answer": response.text.strip()
        }
    except Exception as e:
        logging.error(f"Test endpoint error: {str(e)}")
        return {"error": str(e)}

# Main API endpoint
@app.post("/api/")
async def get_answer(question: str = Form(...), file: UploadFile = None):
    """
    Main API endpoint that accepts a question and optional file
    and returns the answer to IIT Madras graded assignment questions
    """
    try:
        logging.info(f"Received question: {question}")
        
        # Get the model
        model = get_model()
        if not model:
            return JSONResponse(
                content={"error": "Could not initialize AI model"},
                status_code=500
            )
            
        # Process file if uploaded
        if file and file.filename:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, file.filename)
                
                # Save uploaded file
                with open(file_path, "wb") as f:
                    contents = await file.read()
                    f.write(contents)
                
                # Process ZIP files - common in assignments
                if file.filename.endswith('.zip'):
                    extract_dir = os.path.join(temp_dir, "extracted")
                    os.makedirs(extract_dir, exist_ok=True)
                    
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    # Look for CSV files with "answer" column
                    for root, _, files in os.walk(extract_dir):
                        for f in files:
                            if f.endswith('.csv'):
                                csv_path = os.path.join(root, f)
                                df = pd.read_csv(csv_path)
                                
                                if "answer" in df.columns:
                                    answer = str(df["answer"].iloc[0])
                                    logging.info(f"Found answer in CSV: {answer}")
                                    return JSONResponse(content={"answer": answer})
        
        # If no answer found in file, use Gemini AI
        logging.info(f"Generating answer with Gemini AI")
        
        # Format prompt for better results
        prompt = (
            f"You are helping with IIT Madras Online Degree in Data Science assignments.\n\n"
            f"Question: {question}\n\n"
            f"Answer only with the exact answer that should be entered into the assignment form. "
            f"Do not include explanations or anything else. Just the direct answer."
        )
        
        response = model.generate_content(prompt)
        
        if hasattr(response, "text"):
            # Clean up the answer - remove quotation marks, leading/trailing spaces
            answer = response.text.strip()
            answer = answer.replace('"', '').replace("'", "")
            
            # Remove any markdown formatting if present
            if answer.startswith("```") and answer.endswith("```"):
                answer = answer[3:-3].strip()
                
            logging.info(f"Gemini generated answer: {answer}")
            return JSONResponse(content={"answer": answer})
        else:
            raise ValueError("Unexpected response format from Gemini API")

    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error processing request: {error_msg}")
        return JSONResponse(
            content={"error": error_msg},
            status_code=500
        )
