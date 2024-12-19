import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import io  # 추가된 부분

# Streamlit 웹페이지 제목
st.title("[LG이노텍]표준화 회귀 계수 기반 변수 중요도")

# CSV 파일 업로드
st.header('1. 데이터 업로드')
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # 데이터 읽기
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # 변수 선택
    st.header('2. 변수 선택')
    target = st.selectbox('반응변수를 선택하세요.', df.columns)

    # 모든 변수를 선택하고 제외할 변수 선택 기능 추가
    all_features = st.checkbox('모든 변수를 설명변수로 선택')
    if all_features:
        exclude_features = st.multiselect('제외할 변수를 선택하세요.', df.columns.drop(target))
        features = list(set(df.columns.drop(target)) - set(exclude_features))
    else:
        features = st.multiselect('설명변수를 선택하세요.', df.columns.drop(target))

    if target and features:
        # 정규화 수행
        df_z = df.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)

        # 회귀 모델 생성 및 요약 출력
        formula = f"{target} ~ " + " + ".join(features)
        model_z = ols(formula, data=df_z).fit()

        st.subheader("Model Summary")
        st.text(model_z.summary())

        # 회귀 계수 가져오기
        coef_df = pd.DataFrame({
            'Variable': model_z.params.index[1:],  # 첫 번째 인덱스는 상수항이므로 제외
            'Coefficient': model_z.params.values[1:]
        })

        # 절대값 기준 정렬
        coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

        # 바차트 그리기
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if coef < 0 else 'blue' for coef in coef_df['Coefficient']]
        ax.barh(coef_df['Variable'], coef_df['Coefficient'], color=colors)
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Variable')
        ax.set_title('Feature Importance Based on Coefficients')
        ax.invert_yaxis()  # 위에서 아래로 정렬

        # 범례 추가
        red_patch = plt.Line2D([0], [0], color='red', lw=4, label='Negative Coefficient')
        blue_patch = plt.Line2D([0], [0], color='blue', lw=4, label='Positive Coefficient')
        ax.legend(handles=[red_patch, blue_patch])

        # 차트 출력
        st.pyplot(fig)

        # 그림을 버퍼에 저장
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # 다운로드 버튼 추가
        st.download_button(
            label="그림 다운로드",
            data=buf,
            file_name="feature_importance.png",
            mime="image/png"
        )
