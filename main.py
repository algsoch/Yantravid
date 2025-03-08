import os
import sys
import logging
import zipfile
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import tempfile
import json
import datetime
from collections import Counter
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from functools import lru_cache
import time

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

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

# Set up templates and static files with proper directory paths
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Handle the case where static might exist as a file
try:
    # Try to create directories
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        
    if os.path.exists(static_dir) and not os.path.isdir(static_dir):
        os.remove(static_dir)  # Remove if it's a file
        
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        
except Exception as e:
    logging.error(f"Error creating directories: {str(e)}")

# Initialize templates and mount static files (only once!)
templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir, check_dir=False), name="static")

# Log directory information
logging.info(f"Templates directory: {templates_dir}")
logging.info(f"Static directory: {static_dir}")

# Gemini API setup - use environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables")
    # Don't raise an error here - simply log it and let the endpoints fail gracefully

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Global variable for the model and question history
gemini_model = None
question_history = []

# Cached function to get or initialize the model - improves performance
@lru_cache(maxsize=1)
def get_cached_model_name():
    """Return the first working model name"""
    model_names = [
        "models/gemini-1.5-flash",
        "gemini-1.5-flash", 
        "models/gemini-1.5-pro",
        "gemini-1.5-pro"
    ]
    
    for name in model_names:
        try:
            model = genai.GenerativeModel(name)
            # Simple test
            test = model.generate_content("Test")
            if test and hasattr(test, "text"):
                logging.info(f"Found working model: {name}")
                return name
        except Exception as e:
            logging.warning(f"Model {name} failed: {str(e)}")
            continue
            
    return None

# Function to get model - uses cached result
def get_model():
    global gemini_model
    if gemini_model is not None:
        return gemini_model
        
    start_time = time.time()
    model_name = get_cached_model_name()
    
    if not model_name:
        logging.error("No working models found")
        return None
        
    try:
        gemini_model = genai.GenerativeModel(model_name)
        logging.info(f"Model initialized in {time.time() - start_time:.2f}s")
        return gemini_model
    except Exception as e:
        logging.error(f"Error initializing model: {str(e)}")
        return None

# Health check route
@app.get("/")
# Replace the root route with a simpler one

@app.get("/")
async def root(request: Request):
    """Root route that always shows the dashboard"""
    logging.info("Root route accessed")
    return await dashboard(request)
# async def root(request: Request):
#     """Redirect to dashboard or show simple health check"""
#     # If the request accepts HTML, redirect to dashboard
#     if "text/html" in request.headers.get("accept", ""):
#         return RedirectResponse(url="/dashboard")
    
#     # Otherwise return the API status
#     return {
#         "status": "ok", 
#         "service": "IIT Madras Assignment Helper API"
#     }
# Add these debug endpoints

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check environment"""
    return {
        "api_key_exists": bool(GEMINI_API_KEY),
        "question_history": len(question_history),
        "model_initialized": gemini_model is not None,
        "templates_dir_exists": os.path.isdir("templates"),
        "static_dir_exists": os.path.isdir("static")
    }

@app.get("/debug/template")
async def debug_template(request: Request):
    """Test template rendering with minimal data"""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "recent_questions": [{"question": "Test question", "answer": "Test answer", "timestamp": datetime.datetime.now()}],
            "most_frequent": [("Test question", 1)]
        }
    )
# Testing route
@app.get("/test")
async def test():
    """Test endpoint to verify the AI model is working"""
    try:
        start_time = time.time()
        model = get_model()
        if not model:
            return {"error": "Could not initialize AI model"}
        
        logging.info(f"Model retrieved in {time.time() - start_time:.2f}s")
        
        # Generate response
        response = model.generate_content("What is 2+2?")
        logging.info(f"Total test time: {time.time() - start_time:.2f}s")
        
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
    start_time = time.time()
    try:
        logging.info(f"Received question: {question}")
        
        # Get the model
        model = get_model()
        model_time = time.time()
        logging.info(f"Model initialization took: {model_time - start_time:.2f}s")
        
        if not model:
            return JSONResponse(
                content={"error": "Could not initialize AI model"},
                status_code=500
            )
         
        # Process file if uploaded
        answer = None
        if file and file.filename:
            file_start = time.time()
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
                                    logging.info(f"File processing took: {time.time() - file_start:.2f}s")
                                    break
            
            if answer:
                # Record the question and answer
                question_history.append({
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.datetime.now(),
                    "had_file": True
                })
                
                # Limit history size to prevent memory issues
                if len(question_history) > 100:
                    question_history.pop(0)
                    
                logging.info(f"Total request time: {time.time() - start_time:.2f}s")
                return JSONResponse(content={"answer": answer})
        
        # If no answer found in file, use Gemini AI
        logging.info(f"Generating answer with Gemini AI")
        ai_start = time.time()
        
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
            logging.info(f"AI generation took: {time.time() - ai_start:.2f}s")

            # Record the question and answer
            question_history.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.datetime.now(),
                "had_file": file is not None
            })

            # Limit history size to prevent memory issues
            if len(question_history) > 100:
                question_history.pop(0)

            logging.info(f"Total request time: {time.time() - start_time:.2f}s")
            return JSONResponse(content={"answer": answer})
        else:
            raise ValueError("Unexpected response format from Gemini API")

    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error processing request: {error_msg}")
        logging.info(f"Failed request total time: {time.time() - start_time:.2f}s")
        return JSONResponse(
            content={"error": error_msg},
            status_code=500
        )

# Dashboard route
@app.get("/dashboard")
async def dashboard(request: Request):
    """Dashboard showing past questions and a form to ask new ones"""
    start_time = time.time()
    
    # Get the most frequent questions
    question_counts = Counter([q["question"] for q in question_history])
    most_frequent = question_counts.most_common(5)
    
    # Get the most recent questions
    recent_questions = sorted(
        question_history, 
        key=lambda x: x["timestamp"], 
        reverse=True
    )[:10]
    
    logging.info(f"Dashboard rendered in {time.time() - start_time:.2f}s")
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "most_frequent": most_frequent,
            "recent_questions": recent_questions
        }
    )
