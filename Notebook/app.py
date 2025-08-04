import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
# Load model and data
model = joblib.load('C:/Workspaces/Project/DA Elite/June/Notebook/model_pipeline.pkl')
df = pd.read_csv('C:\\Workspaces\\Project\\DA Elite\\June\\Data\\HR_comma_sep.csv')
df.rename(columns={'average_montly_hours': 'average_monthly_hours'}, inplace=True)
df['workload_index'] = df['number_project'] * df['average_monthly_hours']
df['long_term_employee'] = (df['time_spend_company'] >= 4).astype(int)

# Title
st.title("Employee Attrition Dashboard")
st.sidebar.header("Predict Attrition Risk")

def user_input():
    satisfaction = st.sidebar.number_input("Satisfaction Level", 0.0, 1.0)
    evaluation = st.sidebar.number_input("Last Evaluation", 0.0, 1.0, 0.5)
    projects = st.sidebar.number_input("Number of Projects", 1, 10, 3)
    promotion = st.sidebar.selectbox("Promoted in Last 5 Years?", [0, 1])
    department = st.sidebar.selectbox("Department", df['Department'].unique())
    salary = st.sidebar.selectbox("Salary Level", df['salary'].unique())
    hours = st.sidebar.slider("Average Monthly Hours", 90, 310, 160)
    years = st.sidebar.slider("Years at Company", 1, 10, 3)
    accident = st.sidebar.selectbox("Had Work Accident?", [0, 1])
    data = {
        'satisfaction_level': satisfaction,
        'last_evaluation': evaluation,
        'number_project': projects,
        'average_monthly_hours': hours,
        'time_spend_company': years,
        'Work_accident': accident,
        'promotion_last_5years': promotion,
        'Department': department,
        'salary': salary,
        'workload_index': projects * hours,
        'long_term_employee': int(years >= 4)
    }
    return pd.DataFrame([data])

input_df = user_input()
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result")
st.write(f"**Attrition Risk :** {'High Attrition Risk!, ' if prediction == 1 else 'Low Attrition Risk!'}")
st.write(f"There's {proba:.2%} probability that the employee would leave.")
# embed powerbi report on Streamlit app for interactive visualization
components.html(
    '<iframe title="Employee_Attrition" width="600" height="373.5" '
    'src="https://app.powerbi.com/view?r=eyJrIjoiNTg2YTJlZmItMDE1OC00NmY3LWI3NzEtZmNlMmQwMTJiMmM1IiwidCI6I'
    'jk2MmMwNzk0LWNlYWItNDQ4ZS1iNDc0LWIyZjI2MDQ0NmQyMyIsImMiOjEwfQ%3D%3D'
    '&pageName=012e391e4eb45525b62d" frameborder="0" allowFullScreen="true"></iframe>',
    height=400, width=700
)
