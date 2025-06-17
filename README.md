# Fraud Detection System

This project aims to solve the problem of identifying fraudulent bank transactions through the integration of machine learning models and a user-friendly web interface.

## Features

- **Data Preprocessing**: Utilizes targeted feature engineering, including ratio and log-transformed variables, to reveal subtle behavioral patterns in transaction data.
- **Class Imbalance Handling**: Implements SMOTEENN resampling to address class imbalance during model training.
- **Ensemble Machine Learning**:
  - Base learners: Logistic Regression, Random Forest, and Gradient-Boosted Trees.
  - Meta-learner: LightGBM.
  - Optimized using stratified randomized search and average precision (AP) metric.
- **Custom Threshold Optimization**: Balances false positives and negatives via precision-recall analysis.
- **Web Application**:
  - Backend: FastAPI serves a trained model, providing a fraud-prediction score and binary decision results.
  - Frontend: React-based application with an intuitive transaction submission form and real-time results.

## Project Structure

```
.
├── backend
│   ├── main.py                # FastAPI backend implementation
│   ├── requirements.txt       # Python dependencies
│   └── venv                   # Virtual environment (optional)
├── frontend
│   ├── src
│   │   ├── App.js             # Main React component
│   │   ├── index.js           # React entry point
│   │   └── App.css            # Styling for the frontend
│   ├── public
│   │   └── index.html         # Static HTML template
│   ├── README.md              # Frontend-specific README
│   └── package.json           # Node.js dependencies
├── script.sh                  # Script to set up project structure
```

## Getting Started

### Prerequisites

- **Backend**: Python 3.9+ and FastAPI.
- **Frontend**: Node.js and npm.

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will be accessible at `http://localhost:8000`.

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```
   The frontend will be accessible at `http://localhost:3000`.

## Usage

1. Start both the backend and the frontend as described above.
2. Open your browser and navigate to `http://localhost:3000`.
3. Submit transaction attributes through the form to receive a fraud prediction.


## Acknowledgements

Special thanks to publicly available datasets and machine learning libraries like LightGBM, FastAPI, and React for enabling this project.
