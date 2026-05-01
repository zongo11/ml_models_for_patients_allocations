# Repository Guidelines

This repository contains a machine learning system designed to optimize patient orientation to hospitals based on resource availability and proximity. It uses a Random Forest classifier to predict the best hospital allocation.

## Project Structure & Module Organization

The project is structured as a collection of Python scripts and Jupyter notebooks that process patient and hospital data from Excel files.

- **`random_forest_model.py`**: The primary implementation containing data loading, feature engineering, model training, and the allocation system (including a priority queue buffer).
- **`debug_script.py`**: A utility script used for verifying data integrity and resource mapping consistency.
- **`*.ipynb`**: Interactive notebooks used for model exploration and visualization.
- **Datasets**: `Book1.xlsx` (Hospital data) and `patients_1000_ULTRA_COMPLET.xlsx` (Patient data) serve as the primary data sources.

## Build, Test, and Development Commands

The project uses a standard Python environment. A virtual environment is located in `.venv/`.

### Development Commands

- **Run the main model**: `python random_forest_model.py`
- **Run the debug script**: `python debug_script.py`

### Dependencies

Core dependencies include:
- `pandas` & `openpyxl`: Data manipulation and Excel support.
- `scikit-learn`: Machine learning model implementation.
- `numpy`: Numerical operations.

## Coding Style & Naming Conventions

The codebase follows standard Python (PEP 8) conventions:
- **Naming**: `snake_case` for functions and variables, `SCREAMING_SNAKE_CASE` for constants (e.g., `RESOURCE_MAPPINGS`), and `PascalCase` for classes (e.g., `PatientQueue`).
- **Resource Mapping**: High importance is placed on the `RESOURCE_MAPPINGS` dictionary, which bridges French hospital column names with internal system keys.

## System Architecture

The allocation logic follows a two-step process:
1. **Hard Constraints**: Verification of mandatory resources (Lits, Réa, etc.).
2. **ML Prediction**: A Random Forest model scores valid hospitals based on proximity and resource differentials.
3. **Buffer System**: Patients who cannot be immediately allocated are placed in a `PatientQueue` prioritized by ESI (Emergency Severity Index).
