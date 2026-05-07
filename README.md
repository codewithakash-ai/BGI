# Citizen Grievance Management System

Complete local project using **Python + Flask + SQLite**.

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
3. Complaint form UI (Name, Location, Complaint Text, optional photos).
4. Flask submit route:
   - Categorize complaint using rule-based keyword matching
   - Save in DB
   - Return `Complaint registered under [CATEGORY]`
5. Department routing and assignment:
   - Water -> Water Department
   - Electricity -> Electricity Department
   - Roads -> Municipal Department
   - Garbage -> Sanitation Department
   - Healthcare -> Health Department
   - Fire -> Fire Department
6. Admin dashboard:
   - View all complaints
   - Filter by category
   - Update status: Pending -> In Progress -> Resolved
7. Complaint tracking by unique ID (`CMP-YYYYMMDD-XXXXXX`).
8. Analytics dashboard:
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
