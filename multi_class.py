import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

st.title("[LG이노텍]다중 클레스 분류 모델")

# Step 1: 파일 업로드 및 데이터 로드
st.header("1. 데이터 업로드")
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기:", data.head())

    # Step 2: 변수 선택
    st.header('2. 변수 선택')
    target = st.selectbox('반응변수를 선택하세요.', data.columns)
    all_features = st.checkbox('모든 변수를 설명변수로 선택')

    if all_features:
        exclude_features = st.multiselect('제외할 변수를 선택하세요.', data.columns.drop(target))
        features = list(set(data.columns.drop(target)) - set(exclude_features))
    else:
        features = st.multiselect('설명변수를 선택하세요.', data.columns.drop(target))

    if not features:
        st.warning("설명변수를 선택하세요.")
        st.stop()

    # Step 3: 데이터 전처리 및 스케일링
    X = data[features]
    y = data[target]

    # 레이블 인코딩 (필요한 경우)
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Min-Max 스케일링
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Step 4: 모델 학습
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        "LightGBM": LGBMClassifier()
    }

    st.header("3. 모델 학습 및 평가")
    model_accuracies = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        model_accuracies[model_name] = accuracy
        st.subheader(f"{model_name} - 정확도: {accuracy:.4f}")
        st.text(classification_report(y_test, predictions))

    # Step 5: 성능 시각화
    st.header("4. 모델 성능 비교")
    fig, ax = plt.subplots()
    ax.bar(model_accuracies.keys(), model_accuracies.values())
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    for i, v in enumerate(model_accuracies.values()):
        ax.text(i, v, f"{v:.4f}", ha='center')
    st.pyplot(fig)

    # Step 6: 변수 중요도 분석 (Permutation Importance로 한정)
    st.header("5. 변수 중요도 분석 (Permutation Importance)")
    selected_model_name = st.selectbox("분석할 모델 선택", list(models.keys()))
    selected_model = models[selected_model_name]

    # Permutation Importance 계산
    result = permutation_importance(selected_model, X_test, y_test, n_repeats=30, random_state=42)
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': result.importances_mean
    }).sort_values(by='Importance', ascending=False)

    # 변수 중요도 시각화
    st.subheader(f"{selected_model_name} - 변수 중요도")
    fig, ax = plt.subplots()
    ax.bar(feature_importance['Feature'], feature_importance['Importance'])
    ax.set_title(f"{selected_model_name} Feature Importance (Permutation)")
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # 변수 중요도 다운로드 기능
    st.download_button(
        label="변수 중요도 다운로드 (CSV)",
        data=feature_importance.to_csv(index=False),
        file_name='feature_importance.csv',
        mime='text/csv'
    )
