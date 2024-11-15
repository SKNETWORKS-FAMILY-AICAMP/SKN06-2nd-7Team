import streamlit as st
import joblib
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

@st.cache_resource
def load_models():
    with open('best_lr.pkl', 'rb') as f:  # 머신러닝 모델 불러오기 
        ml_model = pickle.load(f)

    input_size = 38
    dl_model = Net(input_size)
    dl_model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))  # 딥러닝 모델 불러오기 
    dl_model.eval()

    return ml_model, dl_model

f_data = pd.read_csv('X_selected.csv')
column_means = f_data.mean()

# 모델 및 전처리기 로드
ml_model, dl_model = load_models()

# Streamlit 앱 설정
st.title('IBM HR 이직 예측 애플리케이션')
st.divider()    # 구분선 삽입


@st.dialog("이직 가능성 예측")
def select_model(item):
    st.write(f"{item}모델을 선택하셨습니다.")
    # EducationField = st.selectbox('전공 학위', ('Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Technical Degree', 'Other'), index=None, placeholder="전공 check...", )
    MaritalStatus = st.selectbox('혼인여부', ("Divorced", "Married", "Single"), index=None, placeholder="상태 check...", )
    NumCompaniesWorked = st.number_input('근무경험', value=None, min_value=0, max_value=15, placeholder="회사를 몇번 옮겼더라...")
    JobRole = st.selectbox('직무', ("Healthcare Representative", "Human Resources", "Laboratory", "Technician", 'Manager',
                                  "Manufacturing Director", "Research Director", "Research", "Scientist", "Sales Executive",
                                  "Sales Representative"), index=None, placeholder="직무 check...", )                                     
    YearsInCurrentRole = st.number_input('근속기간', value=None, min_value=0, max_value=50, placeholder="몇 년 일했더라...")
    MonthlyRate = st.number_input('월급($)', value=None, min_value=2000, max_value=40000, placeholder="2000$ ~ 40000$ 숫자 입력...")
    BusinessTravel = st.selectbox('해외출장 빈도', ("Travel_Frequently", "Travel_Rarely", "Non_Travel"), index=None, placeholder="해외 출장 몇번 갔더라...", )
    OverTime = st.radio("초과근무 여부", ["No", "Yes"])
    JobSatisfaction = st.select_slider("직무 만족도(Level): 숫자가 클 수록 직무에 만족합니다.", options=["1", "2", "3", "4"], value=None)
    EnvironmentSatisfaction = st.select_slider("업무 환경 만족도(Level): 숫자가 클 수록 업무 환경에 만족합니다.", options=["1", "2", "3", "4"], value=None)

    if st.button("예측 결과 조회"):    
        input_data = {
            # f'EducationField_{EducationField}' : 1,    # 사용자 입력값 저장.
            f'MaritalStatus_{MaritalStatus}' : 1,
            'NumCompaniesWorked' : NumCompaniesWorked,
            f'JobRole_{JobRole}': 1,
            'YearsInCurrentRole' : YearsInCurrentRole,
            'MonthlyRate' : MonthlyRate,
            f'BusinessTravel_{BusinessTravel}' : 1,
            'OverTime_Yes' if OverTime == "Yes" else 'OverTime_No': 1,
            'JobSatisfaction': int(JobSatisfaction),
            'EnvironmentSatisfaction': int(EnvironmentSatisfaction),
        }

        def prepare_input(input_data, all_columns, column_means):
            full_input = np.zeros(len(all_columns))
            
            for key, value in input_data.items():
                if key in all_columns:
                    index = all_columns.index(key)  # 해당 키의 인덱스를 찾아서
                    full_input[index] = value       # 해당 인덱스에 값을 채움
        
            # 사용자가 입력하지 않은 값은 column_means를 사용하여 대체
            for i, col in enumerate(all_columns):
                if full_input[i] == 0:  # 입력되지 않은 값이 0이라면
                    if col in column_means:  # column_means에 해당 컬럼이 있으면
                        full_input[i] = column_means[col]  # 해당 평균값으로 채움

            return full_input.reshape(1, -1)  # 모델에 맞게 형상 조정

        # 전체 컬럼 리스트
        all_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'Education',
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

        # 입력 준비
        prepared_input = prepare_input(input_data, all_columns, column_means)

        if item == "딥러닝 모델":
            input_tensor = torch.tensor(prepared_input, dtype=torch.float32)
            prediction = dl_model(input_tensor).item()
        else:
            prediction = ml_model.predict(prepared_input)[0]
        
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
    st.write(f"예측 결과: 이직 가능성은 {prediction_value:.4f}% 으로 {'이직 가능성 있음' if prediction_value > 0.5 else '이직 가능성 낮음'}")

# if st.button("다시 시작"):
#     for key in st.session_state.keys():
#         del st.session_state[key]
#     st.rerun()