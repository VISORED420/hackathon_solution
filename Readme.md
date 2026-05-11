# Resume Matching System using TF-IDF and Cosine Similarity

## Overview
This project implements a simple **Resume-to-Job Description Matching System** using **Natural Language Processing (NLP)** concepts such as **skill normalization, TF-IDF vectorization, and cosine similarity**.

The system compares candidate resumes against multiple job descriptions and ranks the **top 3 most suitable candidates** for each role based on skill similarity.

---

## Features
- Skill normalization using alias mapping
- Duplicate skill removal
- TF-IDF vector generation for resumes
- Binary vector creation for job descriptions
- Cosine similarity calculation
- Automatic ranking of best-matching candidates
- Supports multiple resumes and job descriptions

---

## Technologies Used
- Python 3
- Regular Expressions (`re`)
- Math library (`math`)
- NLP concepts:
  - Text preprocessing
  - Skill normalization
  - TF-IDF
  - Cosine Similarity

---

## Project Structure
```bash
resume-matching/
│
├── main.py              # Main source code
├── README.md            # Project documentation
└── dataset.py           # Resume and job description datasets (optional)