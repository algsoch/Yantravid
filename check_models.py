import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env file!")
    exit(1)

print(f"Using API key: {api_key[:5]}...{api_key[-4:]}")

genai.configure(api_key=api_key)

try:
    # List available models
    print("Available models:")
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            print(f"- {model.name} (can generate content)")
        else:
            print(f"- {model.name}")

    # Try with the fully qualified name
    model = genai.GenerativeModel("models/gemini-1.5-pro")
    response = model.generate_content("What is 2+2?")
    print("\nTest response:", response.text)
    
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    print("\nTry using one of these models instead:")
    print("- gemini-1.5-pro")
    print("- gemini-1.5-flash")
    print("- models/gemini-1.5-pro")