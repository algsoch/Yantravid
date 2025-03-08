import os
import logging
import zipfile
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Setup logging
logging.basicConfig(
    filename="logs/app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Load Gemini API Key from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found! Make sure it's set in .env")
    raise ValueError("GEMINI_API_KEY is missing in .env file!")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

@app.post("/api/")
async def get_answer(question: str = Form(...), file: UploadFile = None):
    try:
        logging.info(f"Received question: {question}")

        # File processing (if provided)
        if file:
            contents = await file.read()
            file_path = f"uploads/{file.filename}"
            os.makedirs("uploads", exist_ok=True)

            with open(file_path, "wb") as f:
                f.write(contents)

            if file.filename.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall("data")

                for root, _, files in os.walk("data"):
                    for f in files:
                        if f.endswith(".csv"):
                            df = pd.read_csv(os.path.join(root, f))
                            answer = str(df["answer"].iloc[0])
                            logging.info(f"Returning extracted answer: {answer}")
                            return JSONResponse(content={"answer": answer}, status_code=200)

        # Generate answer using Gemini AI
        response = gemini_model.generate_content(question)
        if response and hasattr(response, "text"):
            answer = response.text.strip()
            logging.info(f"Returning Gemini-generated answer: {answer}")
            return JSONResponse(content={"answer": answer}, status_code=200)
        else:
            raise Exception("Invalid response from Gemini API")

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
