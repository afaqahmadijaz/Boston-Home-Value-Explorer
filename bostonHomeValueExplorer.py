import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor

# Configure the page
st.set_page_config(page_title="Boston House Price Explorer", layout="wide")

# Title and introduction
st.title("Boston House Price Prediction App")
st.markdown("""
This web application uses a **RandomForestRegressor** to predict the **median value of houses (MEDV)** in the Boston area 
based on user-provided feature values.  
\n
**Instructions**:  
1. Use the left side panel to adjust the feature values.  
2. View your entered parameters in the "Specified Input Parameters" section.  
3. See the predicted house price directly below.  
4. Explore how each feature contributes to the model's prediction in the SHAP plots.  
---
""")

# --- LOAD DATA ---
boston = fetch_openml(name="boston", version=1, as_frame=True)
X = boston.data
Y = boston.target.rename("MEDV")

# Convert any categorical columns to numeric
categorical_cols = X.select_dtypes(["category"]).columns
X[categorical_cols] = X[categorical_cols].apply(lambda col: col.cat.codes)

# Optional: Expandable data preview
with st.expander("Click here for a preview of the dataset (first 5 rows)"):
    st.write(X.head())

# --- SIDEBAR: Input Parameters ---
st.sidebar.header("Specify Input Parameters")
st.sidebar.markdown("""
**Feature Explanations**:
- **CRIM**: Per capita crime rate by town  
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.  
- **INDUS**: Proportion of non-retail business acres per town  
- **CHAS**: Charles River dummy variable (0 = No, 1 = Yes)  
- **NOX**: Nitric oxides concentration (parts per 10 million)  
- **RM**: Average number of rooms per dwelling  
- **AGE**: Proportion of owner-occupied units built before 1940  
- **DIS**: Weighted distances to 5 major Boston employment centers  
- **RAD**: Index of accessibility to radial highways  
- **TAX**: Full-value property-tax rate per $10,000  
- **PTRATIO**: Pupil-teacher ratio by town  
- **B**: 1000(Bk - 0.63)^2, where Bk is the proportion of African Americans by town  
- **LSTAT**: Percentage of lower status of the population  
---
""")

def user_input_features():
    CRIM = st.sidebar.slider(
        "CRIM",
        float(X.CRIM.min()),
        float(X.CRIM.max()),
        float(X.CRIM.mean()),
        help="Per capita crime rate by town"
    )
    ZN = st.sidebar.slider(
        "ZN",
        float(X.ZN.min()),
        float(X.ZN.max()),
        float(X.ZN.mean()),
        help="Proportion of residential land zoned for lots over 25,000 sq.ft."
    )
    INDUS = st.sidebar.slider(
        "INDUS",
        float(X.INDUS.min()),
        float(X.INDUS.max()),
        float(X.INDUS.mean()),
        help="Proportion of non-retail business acres per town"
    )
    CHAS = st.sidebar.selectbox(
        "CHAS (Charles River)",
        [0, 1],
        help="Whether tract bounds the Charles River (1 = Yes, 0 = No)"
    )
    NOX = st.sidebar.slider(
        "NOX",
        float(X.NOX.min()),
        float(X.NOX.max()),
        float(X.NOX.mean()),
        help="Nitric oxides concentration (parts per 10 million)"
    )
    RM = st.sidebar.slider(
        "RM",
        float(X.RM.min()),
        float(X.RM.max()),
        float(X.RM.mean()),
        help="Average number of rooms per dwelling"
    )
    AGE = st.sidebar.slider(
        "AGE",
        float(X.AGE.min()),
        float(X.AGE.max()),
        float(X.AGE.mean()),
        help="Proportion of owner-occupied units built before 1940"
    )
    DIS = st.sidebar.slider(
        "DIS",
        float(X.DIS.min()),
        float(X.DIS.max()),
        float(X.DIS.mean()),
        help="Weighted distances to five Boston employment centers"
    )
    RAD = st.sidebar.slider(
        "RAD",
        float(X.RAD.min()),
        float(X.RAD.max()),
        float(X.RAD.mean()),
        help="Index of accessibility to radial highways"
    )
    TAX = st.sidebar.slider(
        "TAX",
        float(X.TAX.min()),
        float(X.TAX.max()),
        float(X.TAX.mean()),
        help="Full-value property-tax rate per $10,000"
    )
    PTRATIO = st.sidebar.slider(
        "PTRATIO",
        float(X.PTRATIO.min()),
        float(X.PTRATIO.max()),
        float(X.PTRATIO.mean()),
        help="Pupil-teacher ratio by town"
    )
    B = st.sidebar.slider(
        "B",
        float(X.B.min()),
        float(X.B.max()),
        float(X.B.mean()),
        help="1000(Bk - 0.63)^2 where Bk is proportion of African Americans by town"
    )
    LSTAT = st.sidebar.slider(
        "LSTAT",
        float(X.LSTAT.min()),
        float(X.LSTAT.max()),
        float(X.LSTAT.mean()),
        help="Percentage of lower status of the population"
    )

    data = {
        'CRIM': CRIM,
        'ZN': ZN,
        'INDUS': INDUS,
        'CHAS': CHAS,
        'NOX': NOX,
        'RM': RM,
        'AGE': AGE,
        'DIS': DIS,
        'RAD': RAD,
        'TAX': TAX,
        'PTRATIO': PTRATIO,
        'B': B,
        'LSTAT': LSTAT
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# --- Main Panel ---
st.subheader("1. Specified Input Parameters")
st.write("These are the values you’ve chosen in the sidebar:")
st.dataframe(df)

# --- MODEL TRAINING ---
model = RandomForestRegressor(random_state=42)
model.fit(X, Y)  # Y is already a 1D series

# --- PREDICTION ---
prediction = model.predict(df)

# Use columns to format the output
col1, col2 = st.columns([1, 1])  # Two equal-width columns

with col1:
    st.subheader("2. Predicted House Price (MEDV)")
    st.markdown(
        f"""
        <div style="font-size: 1.2rem;">
            <strong>Estimated Price:</strong> {round(prediction[0], 2)}
        </div>
        """, unsafe_allow_html=True
    )

with col2:
    st.subheader("3. Quick Explanation")
    st.write("""
    **MEDV** is the median value of homes in thousands of dollars.  
    This prediction is based on a RandomForestRegressor trained on the Boston dataset.  
    """)

st.markdown("---")

# --- SHAP EXPLANATIONS ---
st.subheader("4. Feature Importance Analysis (SHAP)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# SHAP Summary Plot (Beeswarm)
st.markdown("### SHAP Summary Plot (Beeswarm)")
fig_summary, ax_summary = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig_summary)

st.markdown("---")

# SHAP Bar Plot
st.markdown("### SHAP Bar Plot")
fig_bar, ax_bar = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig_bar)

st.markdown("---")
st.write("**Thank you for using this Boston House Price Prediction App!**")