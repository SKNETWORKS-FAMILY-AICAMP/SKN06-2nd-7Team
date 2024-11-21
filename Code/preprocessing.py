import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from imblearn.over_sampling import SMOTE
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# 데이터 로드 (파일 경로는 필요에 맞게 수정)
data = pd.read_csv('../Data/HR_Employee.csv')

# 범주형 변수 One-Hot Encoding
X = pd.get_dummies(data.drop(columns=['Attrition']), drop_first=True)

y = data['Attrition']

# Label Encoding
le = LabelEncoder()
y = le.fit_transform(y)

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 이상치 처리 - MonthlyIncome 로그 변환
data['MonthlyIncome_Log'] = np.log1p(data['MonthlyIncome'])

# 불필요한 열 제거
columns_to_drop = ['MonthlyIncome', 'EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
data_cleaned = data.drop(columns=columns_to_drop)


# 이상치 처리 - IQR 방식으로 여러 변수 처리
variables_to_process = ['NumCompaniesWorked', 'TrainingTimesLastYear', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

for feature in variables_to_process:
    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    data_cleaned = data_cleaned[(data_cleaned[feature] >= lower_bound) & (data_cleaned[feature] <= upper_bound)]

print(data_cleaned.columns.tolist())

# 수치형 및 범주형 변수 구분 (data_cleaned에서 다시 추출)
numerical_features = data_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = data_cleaned.select_dtypes(include=['object']).columns.tolist()

# 'Attrition'은 타겟 변수이므로 제외
if 'Attrition' in categorical_features:
    categorical_features.remove('Attrition')

print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)

# 범주형 변수와 attrition 간의 카이제곱 검정
for feature in categorical_features:
    contingency_table = pd.crosstab(data_cleaned[feature], data_cleaned['Attrition'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)

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

numerical_features = data_cleaned.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
data_cleaned[numerical_features] = scaler.fit_transform(data_cleaned[numerical_features])

# 수치형 변수 상관계수 기반 제거
numerical_features = data_cleaned.select_dtypes(include=['int64', 'float64']).columns
data_numerical = data_cleaned[numerical_features]

corr_matrix = data_numerical.corr()

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

## 상관계수가 높은 변수 중 하나 제거 (두 번째 변수를 제거)
columns_to_drop = []
for pair in high_corr_pairs:
    columns_to_drop.append(pair[1])

## 중복 제거 및 최종 제거할 수치형 변수 목록 생성
columns_to_drop = list(set(columns_to_drop))

# 카이제곱 검정 기반 범주형 변수 제거
categorical_features = data_cleaned.select_dtypes(include=['object']).columns
categorical_features = categorical_features.drop('Attrition') 

le = LabelEncoder()
y_encoded = le.fit_transform(data_cleaned['Attrition'])

from scipy.stats import chi2_contingency

chi2_insignificant_vars = []

for feature in categorical_features:
    contingency_table = pd.crosstab(data_cleaned[feature], y_encoded)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    if p > 0.05:
        chi2_insignificant_vars.append(feature)

## 최종 제거할 변수 목록에 추가 
columns_to_drop.extend(chi2_insignificant_vars)
columns_to_drop = list(set(columns_to_drop))  # 중복 제거

# 최종 변수 제거
X_selected = data_cleaned.drop(columns=columns_to_drop).drop(columns=['Attrition'])  # 타겟 변수는 제외

# data_cleaned = data_cleaned.drop(columns=columns_to_drop)
# data_cleaned = pd.get_dummies(data_cleaned, drop_first = True)
# X_selected = data_cleaned.drop("Attrition_Yes", axis=1)

X_selected = pd.get_dummies(X_selected, drop_first=True)
X_selected

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size = 0.2, random_state=0)

# # 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
    ('num', numerical_pipeline, numerical_features),  # 수치형 변수 처리
    ('cat', categorical_pipeline, categorical_features)  # 범주형 변수 처리
])

# 최종 파이프라인 (전처리 + 모델)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))  # Logistic Regression 모델 사용
])

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42)

# 모델 학습
pipeline.fit(X_train, y_train)

# 모델 평가 (테스트 세트)
accuracy = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")


# 파이프라인 저장 (joblib 사용)
joblib.dump(preprocessor, '/Users/j/Desktop/AI camp/2차 프로젝트/SKN06-2nd-7Team/Model/preprocessing_pipeline.pkl')
print("Pipeline saved as 'preprocessing_pipeline.pkl'")