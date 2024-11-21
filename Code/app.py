import streamlit as st
import joblib
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
import sys
import os

# 딥러닝 모델 정의
class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lr1 = nn.Linear(input_size, 64)
        self.lr2 = nn.Linear(64, 32)
        self.lr3 = nn.Linear(32, 16)
        self.lr4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.lr1(x))
        x = self.dropout(x)
        x = self.relu(self.lr2(x))
        x = self.dropout(x)
        x = self.relu(self.lr3(x))
        x = self.sigmoid(self.lr4(x))
        return x

# Streamlit 리소스 캐싱
@st.cache_resource
def load_dl_model():
    input_size = 38
    dl_model = Net(input_size)
    dl_model.load_state_dict(torch.load('../Model/best_model.pth', map_location=torch.device('cpu')))
    dl_model.eval()
    return dl_model

@st.cache_resource
def load_ml_pipeline():
    return joblib.load('../Model/ML_pipeline.pkl')

@st.cache_resource
def load_preprocessor():
    sys.path.append(os.getcwd()) # preprocessing.py의 사용자 전처리기들을 load하기위해서 현재 디렉토리를 파이썬 모듈 검색 경로에 추가. 
    return joblib.load('../Model/preprocessing_pipeline.pkl')
    
def load_f_data():
    return pd.read_csv('../Data/f_data.csv')

# 데이터 및 모델 로드
dl_model = load_dl_model()
ml_pipeline = load_ml_pipeline()
preprocessor = load_preprocessor()
f_data = load_f_data()
print(preprocessor)

# 앱 설정
st.title('IBM HR 이직 예측 애플리케이션')
st.divider()

all_columns = ['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
       'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
       'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
       'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']  # 입력 가능한 전체 컬럼 리스트 (생략된 부분 채우기)

set_data = ['Age', 'DailyRate', 'DistanceFromHome', 'Education',
           'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
           'JobSatisfaction', 'MonthlyRate', 'NumCompaniesWorked',
           'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
           'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
           'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
           'YearsSinceLastPromotion', 'YearsWithCurrManager', 'MonthlyIncome_Log',
           'YearsWithOtherCompanies', 'AgeAtJoining', 'IncomePerYearWorked',
           'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
           'JobRole_Human Resources', 'JobRole_Laboratory Technician',
           'JobRole_Manager', 'JobRole_Manufacturing Director',
           'JobRole_Research Director', 'JobRole_Research Scientist',
           'JobRole_Sales Executive', 'JobRole_Sales Representative',
           'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes']


@st.dialog("이직 가능성 예측")
def select_model(item):
    st.write(f"{item} 모델을 선택하셨습니다.")
    # MaritalStatus = st.selectbox('혼인여부', ("Divorced", "Married", "Single"))
    Department = st.selectbox('부서', ['Sales', 'Research & Development', 'Human Resources'])
    NumCompaniesWorked = st.number_input('근무경험', min_value=0, max_value=15)
    JobRole = st.selectbox('직무', ["Healthcare Representative", "Human Resources", "Laboratory", "Technician", 'Manager',
                                  "Manufacturing Director", "Research Director", "Research", "Scientist", "Sales Executive",
                                  "Sales Representative"])  # 가능한 직무 리스트
    # YearsInCurrentRole = st.number_input('근속기간', min_value=0, max_value=50)
    # MonthlyRate = st.number_input('월급($)', min_value=2000, max_value=40000)
    BusinessTravel = st.selectbox('해외출장 빈도', ("Travel_Frequently", "Travel_Rarely", "Non_Travel"))
    OverTime = st.radio("초과근무 여부", ["No", "Yes"])
    JobInvolvement = st.select_slider("직무 몰입도", options=["1", "2", "3", "4"])
    # JobSatisfaction = st.select_slider("직무 만족도", options=["1", "2", "3", "4"])
    # EnvironmentSatisfaction = st.select_slider("업무 환경 만족도", options=["1", "2", "3", "4"])

    if st.button("예측 결과 조회"):
        input_data = pd.DataFrame([[
            # MaritalStatus, NumCompaniesWorked, JobRole, YearsInCurrentRole,
            # MonthlyRate, BusinessTravel, OverTime, JobSatisfaction, EnvironmentSatisfaction
            Department, NumCompaniesWorked, JobRole, BusinessTravel, OverTime, JobInvolvement
        ]], columns=[
            # 'MaritalStatus', 'NumCompaniesWorked', 'YearsInCurrentRole','JobSatisfaction', 'EnvironmentSatisfaction',
                     'Department', 'NumCompaniesWorked', 'JobRole', 'BusinessTravel', 'OverTime', 'JobInvolvement'])

        # f_data를 기준으로 부족한 컬럼 채우기
        for column in all_columns:
            if column not in input_data.columns:
                input_data[column] = f_data[column].iloc[0]

        # 파생 변수 생성
        input_data['MonthlyIncome_Log'] = np.log1p(input_data['MonthlyRate'])
        input_data['YearsWithOtherCompanies'] = (input_data['TotalWorkingYears'] - input_data['YearsAtCompany']).apply(lambda x: max(x, 0))
        input_data['AgeAtJoining'] = (input_data['Age'] - input_data['YearsAtCompany']).apply(lambda x: max(x, 0))
        input_data['IncomePerYearWorked'] = input_data['MonthlyIncome_Log'] / (input_data['TotalWorkingYears'] + 1)

        # 데이터 전처리
        if item == "머신러닝 모델":
            processed_data = ml_pipeline.named_steps['preprocessor'].transform(input_data)
            prediction = ml_pipeline.named_steps['classifier'].predict_proba(processed_data)[:, 1][0]

        elif item == "딥러닝 모델":
            processed_data = preprocessor.transform(input_data)
            processed_data_df = pd.DataFrame(processed_data, columns=preprocessor.get_feature_names_out())
            processed_data_df.columns = processed_data_df.columns.str.replace(r'^num_|^cat_', '', regex=True)
            processed_data_df.columns = processed_data_df.columns.str.lstrip('_')
            processed_data_df.columns
            columns_to_keep = [col for col in processed_data_df.columns if col in set_data]
            processed_data_df = processed_data_df[columns_to_keep]
        
            prediction = dl_model(torch.FloatTensor(processed_data_df.values)).item()

        st.session_state.prediction = prediction
        st.session_state.select_model = {'item': item}
        st.rerun()

if "select_model" not in st.session_state:
    st.write("원하는 모델을 선택해 주세요.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("딥러닝 모델"):
            select_model("딥러닝 모델")
    with col2:
        if st.button("머신러닝 모델"):
            select_model("머신러닝 모델")
else:
    prediction_value = st.session_state.prediction
    st.write(f"예측 결과: {'이직 가능성 있음' if prediction_value > 0.5 else '이직 가능성 낮음'}, 이직 확률: {prediction_value*100:.0f}%")
    if st.button("다시 시작"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
