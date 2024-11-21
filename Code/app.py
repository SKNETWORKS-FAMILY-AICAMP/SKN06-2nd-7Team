import streamlit as st
import time
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

# [Design]
# 상태 초기화
if "page" not in st.session_state:
    st.session_state.page = "main"

# 사이드바 색 변경
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #E0F8E6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

## Side bar
menu = st.sidebar.radio(
    "",
    ("", "Info👩‍💻", "Predictor📱"),
    format_func=lambda x: "Navigation(Select a page)" if x == "" else x,  # 빈 값 표시 변경
    key="menu"
)

## Main Page
if st.session_state.page == "main" and menu == "":
    st.markdown(
        """
          <style>
            .title {
                position: relative;
                top: 160px; 
                text-align: center; 
                font-size: 2.8em;
                color: #0B6121;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<h1 class="title">IBM HR 이직 예측 애플리케이션</h1>', unsafe_allow_html=True)


## Info Page
elif menu == "Info👩‍💻":
    st.markdown(
        """
        <style>
        .title {
            position: relative;
            top: 0px; 
            text-align: center;
            font-size: 1em;
            color: #0B6121;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<h1 class="title">IBM HR 이직 예측 애플리케이션</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .title {
            font-size: 1.8em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.header("**Introduction to Application**")
    st.write("")
    st.write("")
    st.markdown(
        """
         이 애플리케이션은 직원의 이탈 여부를 예측하는 서비스입니다.  
         
         인재 유지는 기업 운영에 있어 중요한 과제 중 하나입니다.  
         직원의 이탈(Attrition)은 조직의 생산성과 안정성에 영향을 미치며,  
         높은 이탈률은 기업 운영에 큰 비용을 초래할 수 있습니다.  
         따라서 직원의 이탈 가능성을 예측하여 사전 대책을 마련하는 것이 필요합니다.  
         
         예측 모델의 구축을 통해 기업이 이탈 가능성이 높은 직원들을 식별하고,  
         사전에 적절한 조치를 취할 수 있도록 도움을 주고자 합니다.  
         본 애플리케이션은 기업이 이탈 가능성이 높은 직원들을 사전에 파악하여,  
         보다 효과적인 인력 관리 전략을 세우는 데 기여할 것입니다.

         ***Stacks***💻
         - Python
         - NumPy
         - Pandas
         - Scikitlearn
         - Streamlit
        """
    )

## Predictor Page
elif menu == "Predictor📱":
    # 상태 초기화
    if "select_model" not in st.session_state:
        st.session_state.select_model = None
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
        
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
                'Executive', 'JobRole_Sales Representative',
                'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes']
        
    # 모델 선택하기
    if  st.session_state.select_model is None:
        st.markdown(
            """
            <style>
            .title {
                position: relative;
                top: 0px; 
                text-align: center;
                font-size: 1em;
                color: #0B6121;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<h1 class="title">IBM HR 이직 예측 애플리케이션</h1>', unsafe_allow_html=True)
        st.header("**Turnover Rate Prediction**")
        st.write("")
        st.write("")
        st.write("🔽 원하는 예측 모델을 선택해 주세요.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Deep Learning Model"):
                st.session_state.select_model = "Deep Learning Model"
        with col2:
            if st.button("Machine Learning Model"):
                st.session_state.select_model = "Machine Learning Model"
        st.stop()

    # 데이터 입력 창
    st.write(f"{st.session_state.select_model} 모델을 선택하셨습니다.")
    st.write("아래 데이터를 입력해주세요.")
    # MaritalStatus = st.selectbox('혼인여부', ("Divorced", "Married", "Single"))
    Department = st.selectbox('부서', ['Sales', 'Research & Development', 'Human Resources'])
    NumCompaniesWorked = st.number_input('근무경험', min_value=0, max_value=15)
    JobRole = st.selectbox('직무', ["Healthcare Representative", "Human Resources", "Laboratory", "Technician", 'Manager', "Manufacturing Director", "Research Director", "Research", "Scientist", "Sales Executive", "Sales Representative"])  # 가능한 직무 리스트
    # YearsInCurrentRole = st.number_input('근속기간', min_value=0, max_value=50)
    # MonthlyRate = st.number_input('월급($)', min_value=2000, max_value=40000)
    BusinessTravel = st.selectbox('해외출장 빈도', ("Travel_Frequently", "Travel_Rarely", "Non_Travel"))
    OverTime = st.radio("초과근무 여부", ["No", "Yes"])
    JobInvolvement = st.select_slider("직무 몰입도", options=["1", "2", "3", "4"])

    if st.button(f"Predict"):
        input_data = pd.DataFrame([[Department, NumCompaniesWorked, JobRole, BusinessTravel, OverTime, JobInvolvement]], columns=['Department', 'NumCompaniesWorked', 'JobRole', 'BusinessTravel', 'OverTime', 'JobInvolvement'])  
    
        # f_data를 기준으로 부족한 컬럼 채우기
        for column in f_data.columns:
            if column not in input_data.columns:
                input_data[column] = f_data[column].iloc[0]

        # 파생 변수 생성
        input_data['MonthlyIncome_Log'] = np.log1p(input_data['MonthlyRate'])
        input_data['YearsWithOtherCompanies'] = (input_data['TotalWorkingYears'] - input_data['YearsAtCompany']).apply(lambda x: max(x, 0))
        input_data['AgeAtJoining'] = (input_data['Age'] - input_data['YearsAtCompany']).apply(lambda x: max(x, 0))
        input_data['IncomePerYearWorked'] = input_data['MonthlyIncome_Log'] / (input_data['TotalWorkingYears'] + 1)

                # 로딩바 생성
        with st.spinner("예측 중..."):
            progress_bar = st.progress(0)
            # 예측 진행 (진행률 업데이트)
            for percent_complete in range(1, 101):
                progress_bar.progress(percent_complete)  # 진행률 업데이트
                time.sleep(0.01)

        # 예측 수행 
        if st.session_state.select_model == "Machine Learning Model":
            processed_data = ml_pipeline.named_steps['preprocessor'].transform(input_data)
            if processed_data.shape[1] == 0:
                st.error("Check data again")
                st.stop()
            prediction = ml_pipeline.named_steps['classifier'].predict_proba(processed_data)[:, 1][0]
        elif st.session_state.select_model == "Deep Learning Model":
            processed_data = preprocessor.transform(input_data)
            processed_data_df = pd.DataFrame(processed_data, columns=preprocessor.get_feature_names_out())
            columns_to_keep = [col for col in set_data if col in processed_data_df.columns]
            processed_data_df = processed_data_df[columns_to_keep]

            # 누락된 컬럼 기본값으로 채우기
            for column in set_data:
                if column not in processed_data_df.columns:
                    processed_data_df[column] = 0  
            
            # 모델 입력 데이터 확인
            if processed_data_df.shape[1] == 0:
                st.error("Check data again")
                st.stop()
            
            # 모델 예측
            prediction = dl_model(torch.FloatTensor(processed_data_df.values)).item()
            
        st.session_state.prediction = prediction
        st.rerun()

    # 예측 결과 출력
    if st.session_state.prediction is not None:
        prediction_value = st.session_state.prediction
        result_text = f"**이직 가능성 있음**" if prediction_value > 0.5 else f"**이직 가능성 낮음**"
        st.markdown(f"Result: {result_text}")
        st.write(f"이직 확률: {prediction_value*100:.0f}%")
        if st.button("Restart"):
            del st.session_state.select_model
            del st.session_state.prediction
            st.rerun()