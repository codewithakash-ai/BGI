# AI-Based Citizen Grievance Classification System

Complete local project using **Python + Flask + SQLite + scikit-learn**.

## Folder Structure

```text
bgi/
├── app.py
├── requirements.txt
├── README.md
├── database/
│   ├── __init__.py
│   ├── db.py
│   └── grievance_system.db        # auto-created on first run
├── models/
│   ├── __init__.py
│   ├── ml_model.py
│   ├── complaint_classifier.pkl   # auto-created on first run
│   └── tfidf_vectorizer.pkl       # auto-created on first run
├── static/
│   └── style.css
└── templates/
    ├── base.html
    ├── index.html
    ├── result.html
    ├── admin.html
    ├── track.html
    └── analytics.html
```

## Features Implemented

1. Clean Flask project layout with modular files.
2. SQLite schema with auto-create:
   - `users`
   - `complaints`
   - `departments`
   - `assignments`
3. ML complaint classifier:
   - `preprocess_text(text)`
   - `train_model()`
   - `classify_complaint(text)`
   - TF-IDF + Logistic Regression + pickle artifacts
4. Complaint form UI (Name, Location, Complaint Text).
5. Flask submit route:
   - Classify complaint
   - Save in DB
   - Return `Complaint registered under [CATEGORY]`
6. Department routing and assignment:
   - Water -> Water Department
   - Electricity -> Electricity Department
   - Roads -> Municipal Department
   - Garbage -> Sanitation Department
   - Healthcare -> Health Department
7. Admin dashboard:
   - View all complaints
   - Filter by category
   - Update status: Pending -> In Progress -> Resolved
8. Complaint tracking by unique ID (`CMP-YYYYMMDD-XXXXXX`).
9. Optional voice input in complaint form (browser speech recognition).
10. Analytics dashboard (Chart.js):
   - Complaints per category
   - Status distribution
   - Most common issue

## Run Locally

1. Create and activate virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Flask app:

```bash
python3 app.py
```

4. Open in browser:

```text
http://127.0.0.1:5000
```

## Web3 Note (Optional)

Current implementation uses SQLite as requested. If you want Web3-backed storage later, keep this Flask app as the frontend/API layer and replace `database/db.py` write operations with smart contract calls while still using the same route/controller flow.
