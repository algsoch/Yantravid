
# IIT Madras Assignment Helper API

An API that helps answer IIT Madras Data Science graded assignment questions.

## Features

- Answer questions from IIT Madras graded assignments
- Process uploaded files (including ZIP files) to extract answers
- Clean, direct responses ready to be entered in the assignment

## API Endpoints

### Health Check
- `GET /`: Returns service status

### Test
- `GET /test`: Tests the AI response with a simple math question

### Main API
- `POST /api/`: Accepts a question and optional file, returns the answer

#### Request Format
