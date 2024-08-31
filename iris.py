import streamlit as st
st.set_page_config(
    page_title="Iris App",
    layout="centered",
    initial_sidebar_state="expanded",
)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
iris = load_iris()
X = iris[['data']]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clfa = RandomForestClassifier()
clfa.fit(X_train, y_train)
def user_input_features():
    st.markdown(
    """
    <style>
    body {
        background-color: red;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    sepal_length = st.sidebar.slider('Sepal length (cm)', float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
    sepal_width = st.sidebar.slider('Sepal width (cm)', float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
    petal_length = st.sidebar.slider('Petal length (cm)', float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
    petal_width = st.sidebar.slider('Petal width (cm)', float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Features')
st.write(df)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    prediction = clfa.predict(df)
    prediction_proba = clfa.predict_proba(df)

st.subheader('Prediction')
st.write(iris.target_names[prediction][0])

st.subheader('Prediction Probability')
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))
