# 1. Import Libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 2. Create Sample Loan Dataset
data = {
    'Age': [25, 35, 45, 20, 30, 40, 50, 23, 34, 42],
    'Income': [50000, 60000, 80000, 20000, 40000, 70000, 90000, 25000, 45000, 75000],
    'LoanAmount': [200, 300, 500, 100, 150, 400, 600, 120, 170, 550],
    'Credit_History': [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
    'Loan_Status': [1, 1, 1, 0, 1, 1, 1, 0, 1, 1]  # 1 = Approved, 0 = Not Approved
}
df = pd.DataFrame(data)

# 3. Split Data
X = df[['Age', 'Income', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Build Streamlit App
st.title("🏦 Loan Approval Prediction App")

st.write("""
### Fill the details to check your Loan Approval Status
""")

# User Inputs
age = st.slider('Age', 18, 70, 30)
income = st.number_input('Monthly Income', min_value=1000, max_value=200000, value=50000, step=1000)
loan_amount = st.number_input('Loan Amount (in thousands)', min_value=10, max_value=1000, value=200, step=10)
credit_history = st.selectbox('Credit History', (1, 0))  # 1 = Good, 0 = Bad

# Predict Button
if st.button('Check Loan Status'):
    user_data = pd.DataFrame([[age, income, loan_amount, credit_history]],
                             columns=['Age', 'Income', 'LoanAmount', 'Credit_History'])
    prediction = model.predict(user_data)[0]
    
    if prediction == 1:
        st.success('✅ Loan Approved!')
    else:
        st.error('❌ Loan Not Approved.')

# Display Model Accuracy
st.write("---")
st.write(f"Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
