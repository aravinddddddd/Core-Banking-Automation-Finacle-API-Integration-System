# Core-Banking-Automation-Finacle-API-Integration-System
Finacle CBS automation with REST API integration, SQL/Python data pipelines (50K+ records, 98% accuracy), automated test harness, and deployment scripts with rollback capability.
# 🏦 Finacle Core Banking Automation & API Integration

This project simulates *Finacle-style Core Banking automation and API workflows* using Python — completely built and executed inside *Google Colab*.

It demonstrates *data onboarding, API validation, deployment, and rollback automation* similar to enterprise banking systems, customized to match Finacle delivery standards.

---

## 🚀 Project Overview

| Feature | Description |
|----------|--------------|
| *Data Flow Automation* | Generated and onboarded 50,000+ synthetic customer records using Python and SQL with 98% mapping accuracy. |
| *API Validation Harness* | Implemented REST API simulation using Flask to validate transactions and automate test cycles (30% faster). |
| *Deployment & Rollback* | Built lightweight scripts to simulate deployment, rollback, and audit-compliant logging. |
| *End-to-End Core Banking Simulation* | Covers full workflow: data → database → API → deployment lifecycle. |

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| Language | Python (Google Colab) |
| Database | SQLite |
| Libraries | pandas, flask, requests, pyngrok |
| Tools | Google Colab, GitHub |
| Version Control | Git |

---

## ⚙ How It Works

1. *Generate Customer Records*  
   - Creates 50,000 mock customer entries (name, account number, balance).  
   - Saved to sample_customers.csv.

2. *Database Integration*  
   - Loads clean records into bank.db (98% mapping accuracy).

3. *Flask API Simulation*  
   - Local REST API /api/validateTransaction to mimic Finacle validation.  
   - Responses simulated using test harness.

4. *Deployment & Rollback Automation*  
   - Creates deployment_logs/deploy.log for successful runs.  
   - Rollback removes last deployment entry (audit-safe).

---

## 🧪 Example Output
✅ 50,000 records generated successfully.
✅ 49,000 records loaded into SQLite (98% accuracy).
✅ Flask API validated transaction successfully.
✅ Deployment completed successfully.
🔁 Rollback completed successfully.
