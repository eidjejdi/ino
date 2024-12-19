
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.title('🧮 회귀 및 분류 분석 웹 애플리케이션')

# 데이터 업로드
st.header('1. 데이터 업로드')
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader('데이터 미리보기')
    st.dataframe(data.head())

    # 결측치 처리
    st.subheader('결측치 처리')
    if st.checkbox('결측치가 있는 행 제거'):
        data.dropna(inplace=True)
        st.write('결측치가 있는 행을 제거했습니다.')

    # 변수 선택
    st.header('2. 변수 선택')
    target = st.selectbox('반응변수를 선택하세요.', data.columns)
    # 모든 변수를 선택하고 제외할 변수 선택 기능 추가
    all_features = st.checkbox('모든 변수를 설명변수로 선택')
    if all_features:
        exclude_features = st.multiselect('제외할 변수를 선택하세요.', data.columns.drop(target))
        features = list(set(data.columns.drop(target)) - set(exclude_features))
    else:
        features = st.multiselect('설명변수를 선택하세요.', data.columns.drop(target))

    if features:
        # 데이터 시각화
        st.header('3. 데이터 시각화')
        if st.checkbox('산점도 행렬 보기'):
            if len(features) <= 5:
                sns.pairplot(data[[target] + features])
                st.pyplot(plt)
            else:
                st.write('변수 개수가 많아 산점도 행렬을 표시할 수 없습니다.')

        # 모델 선택 및 설정
        st.header('4. 모델 선택 및 설정')
        model_type = st.selectbox('모델 유형을 선택하세요.', ['회귀', '분류'])

        if model_type == '회귀':
            regression_type = st.selectbox('회귀 모델을 선택하세요.', ['선형 회귀', 'Decision Tree 회귀', 'Random Forest 회귀'])
            # 하이퍼파라미터 설정
            if regression_type == '선형 회귀':
                loss_function = st.selectbox('손실 함수를 선택하세요.', ['일반 선형 회귀 (OLS)', 'Huber 회귀'])
                if loss_function == 'Huber 회귀':
                    epsilon = st.number_input('Huber epsilon 값', min_value=1.0, max_value=10.0, value=1.35)
            elif regression_type in ['Decision Tree 회귀', 'Random Forest 회귀']:
                auto_max_depth = st.checkbox('Early Stopping을 사용하여 max_depth 결정')
                if auto_max_depth:
                    #if st.button('Early Stopping 실행'):
                    if True:
                        # Early Stopping 실행
                        test_size = 0.2
                        tolerance = 3
                        X_model, X_unused, y_model, y_unused = train_test_split(data[features + [target]], data[target], test_size=0.001, random_state=42)
                        X = X_model[features]
                        y = X_model[target]
                        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
                        best_depth = None
                        if model_type == '회귀':
                            best_score = np.inf
                        else:
                            best_score = 0
                        no_improvement_count = 0

                        for depth in range(1, 20):
                            if regression_type == 'Decision Tree 회귀':
                                temp_model = DecisionTreeRegressor(max_depth=depth)
                            elif regression_type == 'Random Forest 회귀':
                                temp_model = RandomForestRegressor(n_estimators=100, max_depth=depth, n_jobs=-1)

                            temp_model.fit(X_train, y_train)
                            y_pred = temp_model.predict(X_val)
                            score = mean_squared_error(y_val, y_pred)

                            if score < best_score:
                                best_score = score
                                best_depth = depth
                                no_improvement_count = 0
                            else:
                                no_improvement_count += 1

                            if no_improvement_count >= tolerance:
                                #st.write(f"조기 종료: 개선 없이 {tolerance}회 반복, 최적의 max_depth={best_depth}")
                                break
                        else:
                            st.write(f"최적의 max_depth: {best_depth}")

                        st.session_state['best_depth'] = best_depth
                    if 'best_depth' in st.session_state:
                        max_depth = st.number_input('트리의 최대 깊이 (max_depth)', min_value=1, max_value=20, value=st.session_state['best_depth'])
                    else:
                        st.warning('Early Stopping을 먼저 실행해주세요.')
                else:
                    max_depth = st.number_input('트리의 최대 깊이 (max_depth)', min_value=1, max_value=20, value=5)
            else:
                max_depth = None
        else:
            classification_type = st.selectbox('분류 모델을 선택하세요.', ['로지스틱 분류', 'Decision Tree 분류', 'Random Forest 분류'])
            # 하이퍼파라미터 설정
            if classification_type in ['Decision Tree 분류', 'Random Forest 분류']:
                auto_max_depth = st.checkbox('Early Stopping을 사용하여 max_depth 결정')
                if auto_max_depth:
                    #if st.button('Early Stopping 실행'):
                    if True:
                        # Early Stopping 실행
                        test_size = 0.2
                        tolerance = 3
                        X_model, X_unused, y_model, y_unused = train_test_split(data[features + [target]], data[target], test_size=0.001, random_state=42)
                        X = X_model[features]
                        y = X_model[target]
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
                        best_depth = None
                        best_score = 0
                        no_improvement_count = 0

                        for depth in range(1, 20):
                            if classification_type == 'Decision Tree 분류':
                                temp_model = DecisionTreeClassifier(max_depth=depth)
                            elif classification_type == 'Random Forest 분류':
                                temp_model = RandomForestClassifier(n_estimators=100, max_depth=depth, n_jobs=-1)

                            temp_model.fit(X_train, y_train)
                            y_pred = temp_model.predict(X_val)
                            score = accuracy_score(y_val, y_pred)

                            if score > best_score:
                                best_score = score
                                best_depth = depth
                                no_improvement_count = 0
                            else:
                                no_improvement_count += 1

                            if no_improvement_count >= tolerance:
                                #st.write(f"조기 종료: 개선 없이 {tolerance}회 반복, 최적의 max_depth={best_depth}")
                                break
                        else:
                            st.write(f"최적의 max_depth: {best_depth}")

                        st.session_state['best_depth'] = best_depth
                    if 'best_depth' in st.session_state:
                        max_depth = st.number_input('트리의 최대 깊이 (max_depth)', min_value=1, max_value=20, value=st.session_state['best_depth'])
                    else:
                        st.warning('Early Stopping을 먼저 실행해주세요.')
                else:
                    max_depth = st.number_input('트리의 최대 깊이 (max_depth)', min_value=1, max_value=20, value=5)
            else:
                max_depth = None

        # 데이터 분리
        X = data[features]
        y = data[target]

        # 분류 모델의 경우 반응변수 라벨 인코딩
        if model_type == '분류' and classification_type != '로지스틱 분류':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # 모델 학습 및 결과 확인
        st.header('5. 모델 학습 및 결과 확인')
        if st.button('모델 학습 시작'):
            if model_type == '회귀':
                if regression_type == '선형 회귀':
                    X_const = sm.add_constant(X)
                    if loss_function == '일반 선형 회귀 (OLS)':
                        model = sm.OLS(y, X_const).fit()
                    elif loss_function == 'Huber 회귀':
                        model = sm.RLM(y, X_const, M=sm.robust.norms.HuberT(t=epsilon)).fit()

                    st.subheader('모델 결과')
                    st.write(model.summary())

                    predictions = model.predict(X_const)
                    residuals = model.resid

                    # R² 출력
                    st.subheader('모델 평가 지표')
                    st.write('MSE (Mean Squared Error):', mean_squared_error(y, predictions))
                    st.write('MAE (Mean Absolute Error):', mean_absolute_error(y, predictions))
                    st.write('R² (결정계수):', r2_score(y, predictions))

                elif regression_type == 'Decision Tree 회귀':
                    model = DecisionTreeRegressor(max_depth=int(max_depth))
                    model.fit(X, y)
                    predictions = model.predict(X)
                    residuals = y - predictions
                    st.subheader('모델 결과')
                    st.write('변수 중요도:')
                    importance_df = pd.DataFrame({'Features': features, 'Variable Importance': model.feature_importances_})
                    st.write(importance_df)
                    # 평가 지표
                    st.subheader('모델 평가 지표')
                    st.write('MSE (Mean Squared Error):', mean_squared_error(y, predictions))
                    st.write('MAE (Mean Absolute Error):', mean_absolute_error(y, predictions))
                    # 트리 시각화
                    st.subheader('트리 시각화')
                    fig, ax = plt.subplots(figsize=(12, 8))
                    plot_tree(model, feature_names=features, filled=True, ax=ax)
                    st.pyplot(fig)

                elif regression_type == 'Random Forest 회귀':
                    model = RandomForestRegressor(n_estimators=100, max_depth=int(max_depth), n_jobs=-1)
                    model.fit(X, y)
                    predictions = model.predict(X)
                    residuals = y - predictions
                    st.subheader('모델 결과')
                    st.write('변수 중요도:')
                    importance_df = pd.DataFrame({'Features': features, 'Variable Importance': model.feature_importances_})
                    st.write(importance_df)
                    # 평가 지표
                    st.subheader('모델 평가 지표')
                    st.write('MSE (Mean Squared Error):', mean_squared_error(y, predictions))
                    st.write('MAE (Mean Absolute Error):', mean_absolute_error(y, predictions))

            else:
                # 분류 모델
                if classification_type == '로지스틱 분류':
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X, y)
                    predictions = model.predict(X)
                    st.subheader('모델 결과')
                    st.write('회귀 계수:')
                    coef_df = pd.DataFrame({'Features': features, 'Coefficients': model.coef_[0]})
                    st.write(coef_df)
                    st.write('절편 (Intercept):', model.intercept_[0])
                    # 평가 지표
                    st.subheader('모델 평가 지표')
                    st.write('정확도 (Accuracy):', accuracy_score(y, predictions))
                    st.write('분류 리포트:')
                    st.text(classification_report(y, predictions))
                else:
                    model = None
                    if classification_type == 'Decision Tree 분류':
                        model = DecisionTreeClassifier(max_depth=int(max_depth))
                    elif classification_type == 'Random Forest 분류':
                        model = RandomForestClassifier(n_estimators=100, max_depth=int(max_depth), n_jobs=-1)

                    model.fit(X, y)
                    predictions = model.predict(X)
                    st.subheader('모델 결과')
                    st.write('변수 중요도:')
                    importance_df = pd.DataFrame({'Features': features, 'Variable Importance': model.feature_importances_})
                    st.write(importance_df)
                    # 평가 지표
                    st.subheader('모델 평가 지표')
                    st.write('정확도 (Accuracy):', accuracy_score(y, predictions))
                    st.write('분류 리포트:')
                    st.text(classification_report(y, predictions))

                    # 트리 시각화 (Decision Tree의 경우)
                    if classification_type == 'Decision Tree 분류':
                        st.subheader('트리 시각화')
                        fig, ax = plt.subplots(figsize=(12, 8))
                        plot_tree(model, feature_names=features, class_names=le.classes_, filled=True, ax=ax)
                        st.pyplot(fig)

            # 독립변수가 하나인 경우 예측값 시각화
            if model_type == '회귀' and len(features) == 1:
                st.subheader('예측 결과 시각화')
                X_sorted = X.sort_values(by=features[0])
                X_plot = np.linspace(X[features[0]].min(), X[features[0]].max(), 100)
                X_plot_df = pd.DataFrame({features[0]: X_plot})

                if regression_type == '선형 회귀':
                    X_plot_const = sm.add_constant(X_plot_df)
                    y_plot = model.predict(X_plot_const)
                else:
                    y_plot = model.predict(X_plot_df)

                fig, ax = plt.subplots()
                ax.scatter(X[features[0]], y, label='Actual')
                ax.plot(X_plot, y_plot, color='red', label='Predicted')
                ax.set_xlabel(features[0])
                ax.set_ylabel(target)
                ax.legend()
                st.pyplot(fig)

            # 회귀 모델인 경우에만 잔차 플롯 표시
            if model_type == '회귀':
                st.subheader('잔차 플롯')
                fig, ax = plt.subplots()
                ax.scatter(predictions, residuals)
                ax.axhline(0, color='red', linestyle='--')
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                st.pyplot(fig)

            # 결과 저장
            st.header('6. 결과 저장')
            if st.button('모델 결과 다운로드'):
                if model_type == '회귀':
                    if regression_type == '선형 회귀':
                        result_df = model.summary2().tables[1]
                        result_df.to_csv('model_results.csv')
                    else:
                        results = importance_df
                        results.to_csv('model_results.csv', index=False)
                else:
                    results = importance_df if 'importance_df' in locals() else coef_df
                    results.to_csv('model_results.csv', index=False)
                st.write('모델 결과를 model_results.csv 파일로 저장했습니다.')

else:
    st.info('CSV 파일을 업로드해주세요.')
