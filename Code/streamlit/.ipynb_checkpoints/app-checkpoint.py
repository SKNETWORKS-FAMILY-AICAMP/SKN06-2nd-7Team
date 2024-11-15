import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 모델 로드
model = tf.keras.models.load_model('Model/best_model.pth')

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # 범주형 변수 원-핫 인코딩
    categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    
    # 타겟 변수 분리
    X = df_encoded.drop('Attrition', axis=1)
    y = df_encoded['Attrition']
    
    # 특성 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, df

# 메인 함수
def main():
    st.title('HR 이직률 예측 모델 시각화')
    
    # 파일 업로드
    uploaded_file = st.file_uploader("HR 데이터 CSV 파일을 업로드하세요", type="csv")
    
    if uploaded_file is not None:
        X, y, df = load_and_preprocess_data(uploaded_file)
        
        # 예측
        predictions = model.predict(X)
        df['Predicted_Attrition'] = (predictions > 0.5).astype(int)
        
        # 결과 표시
        st.subheader('예측 결과')
        st.write(df[['Age', 'JobRole', 'Attrition', 'Predicted_Attrition']])
        
        # 정확도 계산
        accuracy = np.mean(df['Attrition'] == df['Predicted_Attrition'])
        st.write(f'모델 정확도: {accuracy:.2f}')
        
        # 이직률 시각화
        st.subheader('부서별 이직률')
        dept_attrition = df.groupby('Department')['Predicted_Attrition'].mean()
        fig, ax = plt.subplots()
        dept_attrition.plot(kind='bar', ax=ax)
        plt.title('부서별 예측 이직률')
        plt.ylabel('이직률')
        st.pyplot(fig)
        
        # 특성 중요도 (예시)
        st.subheader('특성 중요도')
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.layers[0].get_weights()[0].sum(axis=1)
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
        fig, ax = plt.subplots()
        feature_importance.plot(x='feature', y='importance', kind='bar', ax=ax)
        plt.title('상위 10개 중요 특성')
        plt.ylabel('중요도')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

if __name__ == '__main__':
    main()