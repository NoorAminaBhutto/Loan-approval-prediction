# 🏦 Loan Approval Prediction - Phase 8 (Final)

**Intern:** Noor Amina Bhutto
**Internship:** BigBrains - Artificial Intelligence and Machine Learning Internship

### 🔗 Project Links
- **GitHub Repository:** https://github.com/NoorAminaBhutto/Loan-approval-prediction
- **Live Deployed App:** https://loan-approval-prediction-kzenw4piqhfufuncxfa7ky.streamlit.app/
- **Demo:** The app predicts APPROVED / REJECTED based on applicant details.

### 1. Problem Statement
Banks spend a lot of time manually verifying loan applications. This project aims to automate the process by predicting loan approval status using machine learning based on applicant's details like income, credit history, and property area.

### 2. Dataset
- **Source:** Loan Prediction Dataset
- **Total Records:** 614 loan applications
- **Input Features:** Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
- **Target Variable:** Loan_Status (Y = Approved, N = Rejected)

### 3. Data Preprocessing
- Handled missing values using Mode for categorical and Median for numerical features.
- Encoded categorical variables using Label Encoding.
- Applied StandardScaler for feature scaling to improve model performance.

### 4. Model Used
- **Model:** Logistic Regression
- **Train-Test Split:** 80% Training, 20% Testing
- **Reason for Selection:** Best for binary classification, fast and interpretable for this dataset.

### 5. Model Metrics & Performance
- **Accuracy:** ~82%
- **Evaluation:** Confusion Matrix and Classification Report
- **Key Insight:** Credit_History is the most important feature. If Credit_History = 1, the chance of approval is >90%.

### 6. Screenshot Proof (For Submission)
1. GitHub repository containing app.py, requirements.txt, README.md
2. Detailed README with documentation
3. Deployed application on Streamlit Community Cloud
4. Final working prediction showing APPROVED/REJECTED result

### 7. Limitations
- The dataset is small (614 rows), a larger dataset would improve accuracy.
- Limited features - CIBIL Score, bank statements, and employment history are not included.
- In future, advanced models like Random Forest and XGBoost can be tested.

### 8. Live Demo
**Deployed Link:** https://loan-approval-prediction-kzenw4piqhfufuncxfa7ky.streamlit.app/

### 9. How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
