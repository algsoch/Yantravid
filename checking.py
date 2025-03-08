import os
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env file")

genai.configure(api_key=GEMINI_API_KEY)

# FastAPI app
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/api/")
async def ask_gemini(question: str = Form(...), file: UploadFile = File(None)):
    try:
        model = genai.GenerativeModel("gemini-pro")  # Ensure model name is correct
        response = model.generate_content(question)

        if not response.text:
            raise HTTPException(status_code=400, detail="No response from Gemini AI")

        return {"answer": response.text}

    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging purpose
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
