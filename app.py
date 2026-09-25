import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv"
df = pd.read_csv(url)
df = df.drop('Loan_ID', axis=1)
for col in ['Gender','Married','Dependents','Self_Employed']:
    df[col] = df[col].fillna(df[col].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

cat_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop('Loan_Status', axis=1)
y = LabelEncoder().fit_transform(df['Loan_Status'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

st.title("🏦 Loan Approval Prediction - Phase 8")

Gender = st.selectbox("Gender", ["Male","Female"])
Married = st.selectbox("Married", ["Yes","No"])
Dependents = st.selectbox("Dependents", ["0","1","2","3+"])
Education = st.selectbox("Education", ["Graduate","Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes","No"])
Property_Area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])
Credit_History = st.selectbox("Credit History (1=Good)", [1,0])
ApplicantIncome = st.number_input("Applicant Income", value=6000)
CoapplicantIncome = st.number_input("Coapplicant Income", value=2000)
LoanAmount = st.number_input("Loan Amount", value=150)
Loan_Amount_Term = st.number_input("Loan Term", value=360)

if st.button("Predict Loan Status"):
    input_data = {'Gender': encoders['Gender'].transform([Gender])[0],'Married': encoders['Married'].transform([Married])[0],'Dependents': encoders['Dependents'].transform([Dependents])[0],'Education': encoders['Education'].transform([Education])[0],'Self_Employed': encoders['Self_Employed'].transform([Self_Employed])[0],'ApplicantIncome': ApplicantIncome,'CoapplicantIncome': CoapplicantIncome,'LoanAmount': LoanAmount,'Loan_Amount_Term': Loan_Amount_Term,'Credit_History': Credit_History,'Property_Area': encoders['Property_Area'].transform([Property_Area])[0]}
    df_input = pd.DataFrame([input_data])
    prob = model.predict_proba(scaler.transform(df_input))[0][1]
    pred = model.predict(scaler.transform(df_input))[0]
    if pred == 1:
        st.success(f"APPROVED ✅ - Probability {prob*100:.2f}%")
    else:
        st.error(f"REJECTED ❌ - Probability {prob*100:.2f}%")
