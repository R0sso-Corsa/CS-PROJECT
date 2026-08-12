# Data Dictionary — Key Tables

This document lists the most important columns for the core database tables. Kept intentionally concise.

**users**

| Column | Type | Constraints | Description | Example |
|---|---|---|---|---|
| id | INT | PK, AUTO_INCREMENT | Primary user identifier | 42 |
| username | VARCHAR(50) | UNIQUE, NOT NULL | Login / display name | demo_user |
| password_hash | VARCHAR(255) | NOT NULL | Bcrypt/argon2 password hash | $2y$... |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | When account was created | 2025-01-01 00:00:00 |

**prediction_jobs**

| Column | Type | Constraints | Description | Example |
|---|---|---|---|---|
| id | INT | PK, AUTO_INCREMENT | Job identifier (queue id) | 1057 |
| user_id | INT | FK -> users(id) | Owner of the job | 42 |
| ticker_symbol | VARCHAR(10) | NOT NULL | Stock symbol to forecast | AAPL |
| status | ENUM | queued/running/completed/failed | Job lifecycle state | running |
| params_json | TEXT | NULLABLE | JSON of hyperparams / options | {"epochs":30} |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | When job was queued | 2025-01-15 14:30:00 |
| completed_at | TIMESTAMP | NULLABLE | When job finished (if any) | 2025-01-15 14:45:00 |

**saved_graphs**

| Column | Type | Constraints | Description | Example |
|---|---|---|---|---|
| id | INT | PK, AUTO_INCREMENT | Graph record id | 88 |
| job_id | INT | FK -> prediction_jobs(id) | Job that produced the graph | 1057 |
| ticker_symbol | VARCHAR(10) | NOT NULL | Associated ticker | AAPL |
| title | VARCHAR(255) | NULLABLE | Human-readable graph title | AAPL Forecast (30d) |
| graph_image | LONGBLOB | BINARY | PNG/JPEG binary image | <binary> |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | When graph was created | 2025-01-15 14:45:00 |

If you want this exported as CSV, JSON or expanded with additional tables/columns, tell me which format or tables to include.
