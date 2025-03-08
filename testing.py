import os
import sys
import unittest
import json
from fastapi.testclient import TestClient
from datetime import datetime
import tempfile
import zipfile
import csv
import pandas as pd
from unittest.mock import patch, MagicMock

# Import your FastAPI app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app, get_model

# Create a test client
client = TestClient(app)

class TestAssignmentHelper(unittest.TestCase):
    
    def test_root_route(self):
        """Test that the root route returns a 200 response"""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
    
    def test_debug_endpoint(self):
        """Test that the debug endpoint returns expected information"""
        response = client.get("/debug")
        data = response.json()
        
        # Check that the response contains expected keys
        self.assertIn("api_key_exists", data)
        self.assertIn("question_history", data)
        self.assertIn("model_initialized", data)
        self.assertIn("templates_dir_exists", data)
        self.assertIn("static_dir_exists", data)
    
    @patch("main.get_model")
    def test_test_endpoint(self, mock_get_model):
        """Test the /test endpoint with a mocked model"""
        # Create a mock model response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "4"
        mock_model.generate_content.return_value = mock_response
        
        # Set up the mock to return our mock model
        mock_get_model.return_value = mock_model
        
        response = client.get("/test")
        data = response.json()
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["question"], "What is 2+2?")
        self.assertEqual(data["answer"], "4")
        
        # Verify the model was called correctly
        mock_model.generate_content.assert_called_once_with("What is 2+2?")
    
    def test_test_endpoint_model_failure(self):
        """Test the /test endpoint when model initialization fails"""
        # Use a context manager to temporarily modify the get_model function
        with patch("main.get_model", return_value=None):
            response = client.get("/test")
            data = response.json()
            
            # Check error response
            self.assertEqual(response.status_code, 200)
            self.assertIn("error", data)
            self.assertEqual(data["error"], "Could not initialize AI model")
    
    @patch("main.get_model")
    def test_api_endpoint_simple_question(self, mock_get_model):
        """Test the API endpoint with a simple question"""
        # Create a mock model response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Paris"
        mock_model.generate_content.return_value = mock_response
        
        # Set up the mock to return our mock model
        mock_get_model.return_value = mock_model
        
        # Test the API endpoint
        response = client.post(
            "/api/",
            data={"question": "What is the capital of France?"}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "Paris")
        
        # Check that the model was called with the right prompt
        args, _ = mock_model.generate_content.call_args
        prompt = args[0]
        self.assertIn("What is the capital of France?", prompt)
    
    def test_api_endpoint_with_file(self):
        """Test the API endpoint with a file upload containing an answer"""
        # Create a temporary CSV file with an answer
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a CSV file with an 'answer' column
            csv_path = os.path.join(temp_dir, "answers.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['answer', 'other_column'])
                writer.writerow(['42', 'some data'])
            
            # Create a ZIP file containing the CSV
            zip_path = os.path.join(temp_dir, "submission.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(csv_path, arcname="answers.csv")
            
            # Open the file for the request
            with open(zip_path, 'rb') as f:
                # Test the API endpoint
                response = client.post(
                    "/api/",
                    data={"question": "What is the answer to life, the universe, and everything?"},
                    files={"file": ("submission.zip", f, "application/zip")}
                )
            
            # Check the response
            if response.status_code == 200:
                data = response.json()
                # Note: This might fail if file processing is disabled
                # In that case, adjust the expected output
                self.assertEqual(data.get("answer"), "42")
    
    def test_dashboard_route(self):
        """Test the dashboard route returns a 200 response"""
        response = client.get("/dashboard")
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    # Create the templates and static directories if they don't exist
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    # Ensure a basic dashboard.html exists
    if not os.path.exists(os.path.join(templates_dir, "dashboard.html")):
        with open(os.path.join(templates_dir, "dashboard.html"), 'w') as f:
            f.write("""<!DOCTYPE html>
            <html>
            <head><title>Dashboard</title></head>
            <body>
                <h1>Dashboard</h1>
                <ul>
                {% for item in recent_questions %}
                    <li>{{ item.question }} - {{ item.answer }}</li>
                {% endfor %}
                </ul>
            </body>
            </html>""")
    
    # Run the tests
    unittest.main()