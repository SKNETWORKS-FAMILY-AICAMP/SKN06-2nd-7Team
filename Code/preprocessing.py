import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib

# 데이터 로드 (파일 경로는 필요에 맞게 수정)
data = pd.read_csv('/Users/j/Desktop/AI camp/2차 프로젝트/SKN06-2nd-7Team/Data/HR_Employee.csv')

# 이상치 처리 - MonthlyIncome 로그 변환
data['MonthlyIncome_Log'] = np.log1p(data['MonthlyIncome'])

# 불필요한 열 제거
columns_to_drop = ['MonthlyIncome', 'EmployeeCount', 'Over18', 'StandardHours']
data_cleaned = data.drop(columns=columns_to_drop)

# 이상치 처리 - IQR 방식으로 여러 변수 처리
variables_to_process = ['NumCompaniesWorked', 'TrainingTimesLastYear', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
for feature in variables_to_process:
    q1 = data_cleaned[feature].quantile(0.25)
    q3 = data_cleaned[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data_cleaned = data_cleaned[(data_cleaned[feature] >= lower_bound) & (data_cleaned[feature] <= upper_bound)]

# Feature Engineering - 새로운 특성 생성
data_cleaned['YearsWithOtherCompanies'] = data_cleaned['TotalWorkingYears'] - data_cleaned['YearsAtCompany']
data_cleaned['YearsWithOtherCompanies'] = data_cleaned['YearsWithOtherCompanies'].apply(lambda x: max(x, 0))

data_cleaned['AgeAtJoining'] = data_cleaned['Age'] - data_cleaned['YearsAtCompany']
data_cleaned['AgeAtJoining'] = data_cleaned['AgeAtJoining'].apply(lambda x: max(x, 0))

data_cleaned['IncomePerYearWorked'] = data_cleaned['MonthlyIncome_Log'] / (data_cleaned['TotalWorkingYears'] + 1)

# 새로운 특성에 대한 이상치 제거 - IQR 방식 적용
new_features = ['YearsWithOtherCompanies', 'AgeAtJoining', 'IncomePerYearWorked']
for feature in new_features:
    q1 = data_cleaned[feature].quantile(0.25)
    q3 = data_cleaned[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data_cleaned = data_cleaned[(data_cleaned[feature] >= lower_bound) & (data_cleaned[feature] <= upper_bound)]

# 타겟 변수와 특성 분리
X = data_cleaned.drop('Attrition', axis=1)
y = data_cleaned['Attrition']

# 수치형 및 범주형 변수 구분
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# 전처리 파이프라인 생성

# 수치형 변수 파이프라인: 결측값 대체 + 표준화
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # 결측값 평균 대체
    ('scaler', StandardScaler())  # 표준화
])

# 범주형 변수 파이프라인: 결측값 대체 + 원-핫 인코딩
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 결측값 최빈값 대체
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 원-핫 인코딩 (알 수 없는 값 무시)
])

# 전체 전처리 파이프라인 구성 (수치형 + 범주형)
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# 최종 파이프라인 (전처리 + 모델)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))  # Logistic Regression 모델 사용
])

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
pipeline.fit(X_train, y_train)

# 모델 평가 (테스트 세트)
accuracy = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# 파이프라인 저장 (joblib 사용)
joblib.dump(pipeline, '/Users/j/Desktop/AI camp/2차 프로젝트/SKN06-2nd-7Team/Model/preprocessing_pipeline.pkl')
print("Pipeline saved as 'preprocessing_pipeline.pkl'")