# AI Talent Intelligence Platform

An AI-powered resume screening and candidate matching platform that automates the initial recruitment screening process by analyzing candidate resumes against job descriptions.

## Overview

The **AI Talent Intelligence Platform** is a web-based recruitment screening system designed to help recruiters evaluate candidates more efficiently.

The platform accepts a candidate's **resume PDF** and a **Job Description**, extracts relevant information, compares the candidate's profile with the job requirements, calculates matching scores, and uses a **Random Forest Machine Learning model** to predict candidate selection.

The application also includes an **AI-based recruiter summary layer** designed to generate a concise summary of the candidate's strengths, weaknesses, missing skills, and hiring recommendation.

## Key Features

- PDF Resume Parsing
- Resume Information Extraction
- Skill Extraction using a predefined skill dictionary
- Job Description Parsing
- Resume–Job Description Skill Matching
- TF-IDF Text Similarity
- Cosine Similarity
- Experience Scoring
- Education Matching
- Weighted Overall Candidate Score
- Random Forest Hiring Prediction
- Selection Probability
- Recruiter-Oriented Summary
- FastAPI Backend
- Interactive Web Dashboard
- Automatic FastAPI Swagger Documentation

## System Architecture

```text
                    Recruiter
                       |
             Resume PDF + Job Description
                       |
                       v
                FastAPI Application
                       |
          +------------+------------+
          |                         |
          v                         v
   Resume Parser                JD Parser
   (pdfplumber)              (Regex/Rules)
          |                         |
          v                         v
 Information Extraction       JD Information
          |                         |
          +------------+------------+
                       |
                       v
                Resume Matcher
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
   Skill Match   TF-IDF + Cosine   Experience &
                 Similarity         Education
        |              |              |
        +--------------+--------------+
                       |
                       v
                 Overall Score
                       |
                       v
              Random Forest Model
                       |
              +--------+--------+
              |                 |
              v                 v
          Selected          Rejected
              |
              v
       Selection Probability
              |
              v
        Recruiter Summary
              |
              v
         Web Dashboard
```

## Application Workflow

### 1. Resume Upload

The recruiter uploads a resume in PDF format through the web interface. The resume is received by the FastAPI `/analyze` endpoint.

### 2. Resume Parsing

The uploaded PDF is processed using **pdfplumber** to extract the raw textual content.

```text
Resume PDF
    ↓
pdfplumber
    ↓
Raw Resume Text
```

### 3. Resume Information Extraction

The extracted resume text is processed to identify:

- Candidate Name
- Email
- Phone Number
- Skills
- Education
- Experience

### 4. Job Description Parsing

The entered Job Description is processed to extract:

- Required Skills
- Education Requirements
- Experience Requirements

### 5. Skill Matching

The candidate's skills are compared with the skills extracted from the Job Description.

```text
Skill Match Percentage =
Matched Skills / Required JD Skills × 100
```

The system also identifies matched and missing skills.

### 6. TF-IDF Text Similarity

TF-IDF (Term Frequency–Inverse Document Frequency) converts the resume and Job Description into numerical vectors based on word importance.

**Cosine Similarity** then compares the two vectors.

A value closer to `1` indicates greater textual similarity, while a value closer to `0` indicates lower similarity.

### 7. Experience Score

The candidate's experience is compared with the experience required by the Job Description.

If the candidate meets or exceeds the requirement:

```text
Experience Score = 100%
```

Otherwise, a proportional score is calculated.

### 8. Education Score

The platform compares education information extracted from the resume and Job Description.

If a matching education requirement is found:

```text
Education Score = 100%
```

Otherwise:

```text
Education Score = 0%
```

### 9. Overall Candidate Score

The four scores are combined using weighted scoring:

| Component | Weight |
|---|---:|
| Skill Match | 45% |
| Text Similarity | 30% |
| Experience | 15% |
| Education | 10% |

```text
Overall Score =
    (Skill Score × 0.45)
  + (Similarity Score × 0.30)
  + (Experience Score × 0.15)
  + (Education Score × 0.10)
```

## Machine Learning Component

### Random Forest Classifier

The platform uses a **Random Forest Classifier** from Scikit-learn for candidate hiring prediction.

### Features

```text
1. Skill Score
2. Similarity Score
3. Experience Score
4. Education Score
```

### Target

```text
hired

1 → Selected
0 → Rejected
```

Training data is stored in:

```text
data/model/training_data.csv
```

The model uses `predict_proba()` to obtain the predicted probability for the Selected class.

> **Note:** The current project uses a sample labeled training dataset for demonstration. A production system would require a larger historical recruitment dataset with validated hiring outcomes.

## AI Recruiter Summary

The project includes an AI summary layer designed to generate a recruiter-friendly interpretation of the candidate analysis.

The summary can use:

- Candidate skills
- Required JD skills
- Matched skills
- Missing skills
- Education
- Experience
- Skill match percentage
- ML prediction
- Selection probability

The intended output contains:

- Candidate strengths
- Weaknesses
- Missing skills
- Hiring recommendation

A fallback recruiter-summary implementation is also included so the application can continue functioning when an external AI service is unavailable.

## FastAPI Backend

**FastAPI** is used as the backend framework connecting the web interface with the resume processing, matching, machine learning, and summary-generation components.

### Main Routes

#### `GET /`

Loads the main web application.

#### `POST /analyze`

Accepts:

- Resume PDF
- Job Description

and executes the complete candidate analysis pipeline.

## API Documentation

FastAPI automatically provides interactive API documentation through Swagger UI.

After starting the application, open:

```text
http://127.0.0.1:8000/docs
```

Main analysis endpoint:

```text
POST /analyze
```

## Technology Stack

### Backend

- Python
- FastAPI
- Jinja2

### Resume Processing

- pdfplumber

### NLP / Text Processing

- Regular Expressions
- TF-IDF
- Cosine Similarity
- Predefined Skill Dictionary

### Machine Learning

- Scikit-learn
- Random Forest Classifier
- Pandas

### Frontend

- HTML
- CSS
- Jinja2 Templates

### AI

- Google Gemini API integration
- Rule-based recruiter summary fallback

### Version Control

- Git
- GitHub

## Project Structure

```text
AI-Talent-Intelligence-Platform/
│
├── data/
│   ├── job_descriptions/
│   ├── model/
│   │   └── training_data.csv
│   └── skills/
│       └── skills.txt
│
├── src/
│   ├── __init__.py
│   ├── ai_summary.py
│   ├── gemini_summary.py
│   ├── information_extractor.py
│   ├── jd_parser.py
│   ├── matcher.py
│   ├── ml_model.py
│   └── resume_parser.py
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── test_jd_parser.py
├── test_matcher.py
├── test_ml_model.py
├── test_parser.py
└── test_summary.py
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-github-username>/AI-Talent-Intelligence-Platform.git
```

### 2. Navigate to the Project

```bash
cd AI-Talent-Intelligence-Platform
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

If using the Gemini integration, create a `.env` file in the project root:

```text
GOOGLE_API_KEY=your_api_key_here
```

**Never commit your `.env` file or API key to GitHub.**

The `.gitignore` file excludes `.env`.

## Running the Application

Start the FastAPI server from the project root:

```bash
uvicorn main:app
```

The application will run at:

```text
http://127.0.0.1:8000
```

Open the URL in your browser, upload a resume, enter a Job Description, and click **Analyze Resume**.

## Testing

The project includes test files for major components:

```text
test_parser.py
test_jd_parser.py
test_matcher.py
test_ml_model.py
test_summary.py
```

Individual tests can be executed using:

```bash
python test_parser.py
python test_jd_parser.py
python test_matcher.py
python test_ml_model.py
python test_summary.py
```

## Example Output

```text
Skill Match: 70.59%

Matched Skills:
- Python
- SQL
- FastAPI
- Git

Missing Skills:
- Docker
- Kubernetes

Similarity Score: 55.32%
Experience Score: 100%
Education Score: 100%

Overall Score: 73.14%

Prediction: Selected
Selection Probability: 82%
```

The actual values depend on the uploaded resume and Job Description.

## Security

The following files and directories are excluded from version control:

```text
.env
.venv/
__pycache__/
data/processed/
*.log
```

Uploaded and processed resumes are not intended to be stored in the GitHub repository.

## Future Improvements

- Train the model using a larger real-world recruitment dataset
- Improve skill normalization and synonym handling
- Improve education normalization
- Add multiple resume ranking
- Add recruiter analytics and visualizations
- Improve semantic matching using modern embedding models
- Improve LLM-based recruiter summary generation
- Add model evaluation using precision, recall, and F1-score
- Add model versioning
- Deploy the application as a live web service
- Add persistent cloud storage for uploaded resumes
- Add authentication and recruiter accounts

## Project Objective

The primary objective is to reduce the manual effort involved in initial candidate screening by combining:

```text
Resume Parsing
      +
NLP-Based Matching
      +
Machine Learning
      +
AI-Assisted Summarization
      +
Web Application
```

into a single recruitment intelligence platform.

## Project Information

**Project:** AI Talent Intelligence Platform for Intelligent Resume Screening & Candidate Matching

**Qualification:** PG-Diploma (CDAC)

**Technologies:** Python, FastAPI, Scikit-learn, NLP, Machine Learning, pdfplumber, Jinja2, HTML, CSS, Git, GitHub

**Repository:**

```text
https://github.com/abdul-aman2910/AI-Talent-Intelligence-Platform
```

## License

This project is developed for educational and demonstration purposes.
