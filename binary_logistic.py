import streamlit as st
import numpy as np
import pandas as pd
import os
import shutil
import zipfile
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
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
# Set random seed for reproducibility
np.random.seed(0)


# 이미지 전처리 함수
def preprocess_images(folder_path):
    data = []
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        with Image.open(img_path) as img:  # 파일이 자동으로 닫히도록 함
            img = img.convert("L").resize((28, 28))
            data.append(np.array(img).flatten())
    return np.array(data)


# 평가 결과 출력 함수
def print_score(clf, X_train, y_train, X_test, y_test):
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)

    train_accuracy = accuracy_score(y_train, pred_train)
    test_accuracy = accuracy_score(y_test, pred_test)

    train_report = pd.DataFrame(classification_report(y_train, pred_train, output_dict=True))
    test_report = pd.DataFrame(classification_report(y_test, pred_test, output_dict=True))

    st.write("### 훈련 결과")
    st.write(f"정확도: {train_accuracy:.2%}")
    st.write("혼동 행렬:")
    plot_confusion_matrix(confusion_matrix(y_train, pred_train), ["NG", "OK"])
    st.write("분류 리포트:")
    st.write(train_report)

    st.write("### 테스트 결과")
    st.write(f"정확도: {test_accuracy:.2%}")
    st.write("혼동 행렬:")
    plot_confusion_matrix(confusion_matrix(y_test, pred_test), ["NG", "OK"])
    st.write("분류 리포트:")
    st.write(test_report)


# 혼동 행렬 그리기 함수
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"size": 24, "weight": "bold"})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)


# Streamlit UI
st.title("[LG이노텍]이미지 분류를 위한 로지스틱 모델")

# 각 클래스별 ZIP 파일 업로드
ng_zip = st.file_uploader("NG 클래스 이미지 ZIP 파일 업로드", type="zip")
ok_zip = st.file_uploader("OK 클래스 이미지 ZIP 파일 업로드", type="zip")

# PCA 선택
apply_pca = st.checkbox("PCA 적용")
if apply_pca:
    n_components = st.slider("PCA 주성분 수 선택", 10, 100, 70)

# 오버샘플링 옵션
oversample = st.checkbox("Random OverSampling 적용")

# 모델 선택
model_type = st.selectbox("모델 선택", ["Logistic Regression", "Gaussian Naive Bayes", "Decision Tree"])

# Run 버튼
if st.button("Run"):
    if ng_zip and ok_zip:
        # 디렉토리 생성
        if not os.path.exists("uploaded_images/NG"):
            os.makedirs("uploaded_images/NG")
        if not os.path.exists("uploaded_images/OK"):
            os.makedirs("uploaded_images/OK")

        # NG 클래스 이미지 추출
        with zipfile.ZipFile(ng_zip, 'r') as zip_ref:
            zip_ref.extractall("uploaded_images/NG")
        NG_data = preprocess_images("uploaded_images/NG")
        NG_labels = np.ones(len(NG_data))

        # OK 클래스 이미지 추출
        with zipfile.ZipFile(ok_zip, 'r') as zip_ref:
            zip_ref.extractall("uploaded_images/OK")
        OK_data = preprocess_images("uploaded_images/OK")
        OK_labels = np.zeros(len(OK_data))

        # 데이터 및 레이블 결합
        data = np.concatenate((NG_data, OK_data), axis=0)
        labels = np.concatenate((NG_labels, OK_labels), axis=0)

        # 데이터 정규화
        data = data / 255.0

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0)

        # PCA 적용
        if apply_pca:
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # 오버샘플링 적용
        if oversample:
            sampler = RandomOverSampler(random_state=0)
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        # 모델 선택 및 학습
        if model_type == "Logistic Regression":
            #model = LogisticRegression(C=0.1, penalty='l1', solver='saga', max_iter=1000)
            model = LogisticRegression()
        elif model_type == "Gaussian Naive Bayes":
            model = GaussianNB()
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier()

        # 모델 훈련
        model.fit(X_train, y_train)
        print_score(model, X_train, y_train, X_test, y_test)

        # 처리 후 파일 삭제
        if os.path.exists("uploaded_images"):
            shutil.rmtree("uploaded_images")
    else:
        st.warning("NG와 OK ZIP 파일을 모두 업로드해주세요.")

