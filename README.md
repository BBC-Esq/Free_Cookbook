# 🍳 Cookbook

A desktop recipe manager built with Python and PySide6.

## ✨ Features

- Browse and search 59 recipes across 8 categories
- Filter by category or search by name
- View detailed recipe info: ingredients, instructions, prep/cook times, and servings
- Select multiple recipes to generate a consolidated shopping list with merged ingredients
- Export individual recipe ingredients or shopping lists to PDF

## 🚀 Getting Started

### Requirements

- Python 3.10+

### Installation

```bash
pip install -r requirements.txt
```

### Running

```bash
python main.py
```

The database is created automatically on first launch — no setup needed.

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Application entry point and GUI |
| `seed_data.py` | Recipe data used to populate the database |
| `requirements.txt` | Python dependencies |
| `cookbook.db` | SQLite database (auto-generated) |

## 🛠️ Built With

- [PySide6](https://doc.qt.io/qtforpython-6/) — Qt for Python (GUI)
- [ReportLab](https://www.reportlab.com/) — PDF generation
- SQLite — Local database
