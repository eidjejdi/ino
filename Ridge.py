import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform

# 시스템에 맞는 폰트 설정
if platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
elif platform.system() == 'Darwin':  # MacOS
    font_name = 'AppleGothic'
else:
    font_name = 'NanumGothic'  # Linux 환경 등에서 사용할 수 있는 폰트

rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 앱 구성
st.title('[LG이노텍]벌점 회귀 분석')

# 데이터 업로드
st.header('1. 데이터 업로드')
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("업로드한 데이터:")
    st.write(data)

    # 변수 선택
    st.header('2. 변수 선택')
    target = st.selectbox('반응변수를 선택하세요.', data.columns)
    all_features = st.checkbox('모든 변수를 설명변수로 선택')
    if all_features:
        exclude_features = st.multiselect('제외할 변수를 선택하세요.', data.columns.drop(target))
        features = list(set(data.columns.drop(target)) - set(exclude_features))
    else:
        features = st.multiselect('설명변수를 선택하세요.', data.columns.drop(target))

    if target and features:
        X = data[features]
        y = data[[target]]

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # 회귀 분석 선택
        st.header('3. 회귀 분석 선택')
        regression_type = st.selectbox('회귀 유형을 선택하세요.', ['선형 회귀', '2차 다항 회귀'])

        # 모델 선택 (OLS 추가 및 위치 변경)
        st.subheader('모델 선택')
        model_type = st.selectbox('모델 유형을 선택하세요.', ['OLS', 'Ridge', 'Lasso', 'ElasticNet'])

        if st.button('Run'):
            # 데이터 전처리
            if regression_type == '선형 회귀':
                # 선형 회귀의 경우 원본 데이터 사용
                X_train_processed = X_train
                X_test_processed = X_test
                feature_names = X.columns
            elif regression_type == '2차 다항 회귀':
                # 2차 다항 회귀의 경우 다항 특성 생성
                poly = PolynomialFeatures(degree=2, include_bias=False)
                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                X_train_processed = poly.fit_transform(X_train_scaled)
                X_test_processed = poly.transform(X_test_scaled)
                feature_names = poly.get_feature_names_out(input_features=X.columns)
            else:
                st.error('올바른 회귀 유형을 선택하세요.')
                st.stop()

            # 모델 선택 및 학습
            if model_type == 'OLS':
                model = LinearRegression()
            elif model_type == 'Ridge':
                model = Ridge(alpha=0.2)
            elif model_type == 'Lasso':
                model = Lasso(alpha=0.001, max_iter=10000)
            elif model_type == 'ElasticNet':
                model = ElasticNetCV(alphas=np.logspace(-4, 0, 100),
                                     l1_ratio=np.arange(0.01, 1, 0.1),
                                     max_iter=3000, n_jobs=-1)
            else:
                st.error('올바른 모델 유형을 선택하세요.')
                st.stop()

            # 모델 학습
            model.fit(X_train_processed, y_train)
            y_pred = model.predict(X_test_processed)
            y_train_pred = model.predict(X_train_processed)

            # 결과 출력
            st.subheader(f'{model_type} 회귀 결과')

            # 회귀 계수 출력
            coefficients = pd.DataFrame({
                '변수명': feature_names,
                '회귀 계수': model.coef_.flatten()
            })
            st.write(coefficients)
            st.write("절편:", model.intercept_)

            st.write("R^2 (훈련 세트):", r2_score(y_train, y_train_pred))
            st.write("R^2 (테스트 세트):", r2_score(y_test, y_pred))
            st.write("MSE (훈련 세트):", mean_squared_error(y_train, y_train_pred))
            st.write("MSE (테스트 세트):", mean_squared_error(y_test, y_pred))

            # 시각화
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel('실제 값')
            ax.set_ylabel('예측 값')
            ax.set_title(f'{model_type} 회귀 예측 결과')
            st.pyplot(fig)
