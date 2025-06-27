import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

# --- Set Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Define Feature Transformers ---
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Applies log transformation to specified columns."""
        X = pd.DataFrame(X)
        if 'NumOfProducts' in X.columns:
            X['NumOfProducts_log'] = np.log1p(X['NumOfProducts'])
            X.drop(columns=['NumOfProducts'], inplace=True)
        if 'Age' in X.columns:
            X['Age_log'] = np.log1p(X['Age'])
            X.drop(columns=['Age'], inplace=True)
        return X

class GenderLocationBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Converts 'Gender' and 'Location' into binary columns."""
        X = pd.DataFrame(X)
        if 'Gender' in X.columns:
            X['Gender_Female'] = (X['Gender'] == 'Female').astype(int)
            X['Gender_Male'] = (X['Gender'] == 'Male').astype(int)
            X.drop(columns=['Gender'], inplace=True)
        if 'Location' in X.columns:
            X['Location_France'] = (X['Location'] == 'France').astype(int)
            X['Location_Germany'] = (X['Location'] == 'Germany').astype(int)
            X['Location_Spain'] = (X['Location'] == 'Spain').astype(int)
            X.drop(columns=['Location'], inplace=True)
        return X

def map_yes_no(df):
    """Maps 'Yes'/'No' values to 1/0 for categorical columns."""
    if isinstance(df, pd.DataFrame):
        return df.applymap(lambda x: {'Yes': 1, 'No': 0}.get(x, x))
    elif isinstance(df, pd.Series):
        return df.map({'Yes': 1, 'No': 0})
    else:
        raise TypeError("Expected input to be a DataFrame or Series.")

# --- Load Model and Pipeline ---
try:
    with open('pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
except AttributeError as e:
    st.error(f"Error loading pipeline or model: {e}")
    
# Prediction function
def make_prediction(input_data):
    try:
        # Convert input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Process the data through the pipeline (transformation + prediction)
        processed_data = pipeline.transform(input_df)

        # Make predictions using the model
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0, 1]

        return "Churn" if prediction == 1 else "No Churn", probability
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None
    
st.markdown("""
    <style>
    .kpi-row {
        display: flex;
        justify-content: space-between;
        gap:15px;
        margin-bottom: 15px;
    }
    .kpi-box {
        background-color: #24476C;
        border-radius: 10px;
        text-align: center;
        font-size: 19px;
        padding: 9px;
        flex: 1; 
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
        line-height: 0.6;
    }
    .custom-subheader {
        font-size: 27px;
        font-weight: 650;
        background: linear-gradient(60deg, #c1e8ff, #0033ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
    }
    .title {
        background: linear-gradient(60deg, #c1e8ff, #0033ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 45px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

## CODE FOR DASHBOARD
def overview_page():    # Function for the overview page

    st.markdown("""
        <div class="kpi-row">
            <div class="kpi-box">
                <strong>Total Analyzed</strong><br>
                <span style="color:#EFA00F; font-size:24px;">10,000</span>
            </div>
            <div class="kpi-box">
                <strong>Model Accuracy</strong><br>
                <span style="color:#EFA00F; font-size:24px;">86 %</span>
            </div>
            <div class="kpi-box">
                <strong>Churn Rate</strong><br>
                <span style="color:#EFA00F; font-size:24px;">21.5 %</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("images/age.png", caption="Overview of Age", use_container_width=True)

    with col2:
        st.image("images/creditscore.png", caption="Overview of Credit Score", use_container_width=True)

    with col3:
        st.image("images/estimatedsalary.png", caption="Overview of Estimated Salary", use_container_width=True)
        
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.image("images/balance.png", caption="Overview of Balance", use_container_width=True)

    with col5:
        st.image("images/noofproducts.png", caption="Overview of Number of Products", use_container_width=True)
    
    with col6:
        st.image("images/tenure.png", caption="Overview of Tenure", use_container_width=True)
    
def performance_analysis_page():    # Function for the model performance page

    st.markdown("""
        <div class="kpi-row">
            <div class="kpi-box">
                <strong>Accuracy</strong><br>
                <span style="color:#EFA00F; font-size:24px;">86 %</span>
            </div>
            <div class="kpi-box">
                <strong>Precision</strong><br>
                <span style="color:#EFA00F; font-size:24px;">70 %</span>
            </div>
            <div class="kpi-box">
                <strong>F1 Score</strong><br>
                <span style="color:#EFA00F; font-size:24px;">60 %</span>
            </div>
            <div class="kpi-box">
                <strong>ROC-AUC score</strong><br>
                <span style="color:#EFA00F; font-size:24px;">74 %</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 =st.columns([1.5,1.53])
    
    with col1:
        st.image("images/feature_importance.png", use_container_width=True)
    with col2:
        st.image("images/churn trend.png", use_container_width=True)

def future_prediction_page():
    
    st.markdown("""
    <div class="custom-subheader">Future Prediction Insights:</div>
    """, unsafe_allow_html=True)
    
    st.write("These visualizations highlight how variations in key factors impact customer churn over time. By adjusting column values, helping to refine retention strategies.")
    
    col1, col2, col3= st.columns(3)
    
    with col1:
        st.image("images/fp_age.png", caption="Overview of Age", use_container_width=True)
    with col2:
        st.image("images/fp_creditscore.png", caption="Overview of Credit Score", use_container_width=True)
    with col3:
        st.image("images/fp_salary.png", caption="Overview of Estimated Salary", use_container_width=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.image("images/fp_balance.png", caption="Overview of balance", use_container_width=True)
    with col5:
        st.image("images/fp_nop.png", caption="Overview of Number of products", use_container_width=True)
    with col6:
        st.image("images/fp_tenure.png",caption="Overview of Tenure", use_container_width=True)

def sidebar_churn_prediction():     # Sidebar form for predicting customer churn and switching to Report Page
    
    st.sidebar.markdown("""
    <div style="text-align: center; margin-top: -20px; color: #c1e8ff; font-size: 16px;">
        <p>Enter the customer details below:</p>
    </div>
    """, unsafe_allow_html=True)

    # Ensure proper session state initialization
    if "customer_profile" not in st.session_state:
        st.session_state["customer_profile"] = {}
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = (None, None)  # Safer default
    if "feature_importance" not in st.session_state:
        st.session_state["feature_importance"] = np.zeros(10)  # Default size
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Prediction"  # Default view
    if "pdf_report" not in st.session_state:
        st.session_state["pdf_report"] = None  # Default PDF state

    with st.sidebar.form("churn_prediction_form", clear_on_submit=False):
        credit_score = st.slider("Credit Score", 300, 850, 600)
        tenure = st.slider("Tenure (Years)", 0, 10, 3)
        num_of_products = st.slider("Number of Products", 1, 10, 2)
        age = st.slider("Age", 18, 100, 35)
        balance = st.number_input("Amount Balance ($)", min_value=0, max_value=200000, value=10000, step=500)
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0, max_value=200000, value=50000, step=1000)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        location = st.selectbox("Location", ["France", "Germany", "Spain"], index=0)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"], index=0)
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"], index=0)
        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        # Store customer profile in session state
        st.session_state["customer_profile"] = {
            "CreditScore": credit_score,
            "Tenure": tenure,
            "Balance": balance,
            "EstimatedSalary": estimated_salary,
            "NumOfProducts": num_of_products,
            "Age": age,
            "Gender": gender,
            "Location": location,
            "HasCrCard": 1 if has_cr_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active_member == "Yes" else 0
        }

        # Run prediction
        prediction_result = make_prediction(st.session_state["customer_profile"])

        if prediction_result is None:
            st.session_state["prediction_result"] = (None, None)
            st.session_state["feature_importance"] = np.zeros(len(st.session_state["customer_profile"]))
        else:
            if len(prediction_result) == 3:
                prediction, probability, feature_importance = prediction_result
            else:
                prediction, probability = prediction_result
                feature_importance = np.zeros(len(st.session_state["customer_profile"]))

            # Store results before rerunning
            st.session_state["prediction_result"] = (prediction, probability)
            st.session_state["feature_importance"] = feature_importance
            
            # Generate PDF **automatically** upon prediction
            pdf_file = generate_pdf()
            st.session_state["pdf_report"] = pdf_file  # Store PDF in session state
        
    # Safely retrieve results AFTER state refresh
    prediction, probability = st.session_state.get("prediction_result", (None, None))

    if prediction is not None and probability is not None:
        if probability < 0.5:
            st.sidebar.success(f"✅ Low Risk: {probability*100:.2f}% chance of churn.")
        elif 0.5 < probability < 0.7:
            st.sidebar.warning(f"⚠ Medium Risk: {probability*100:.2f}% chance of churn.")
        else:
            st.sidebar.error(f"⚠ High Risk: {probability*100:.2f}% chance of churn.")
            
    

def get_retention_suggestion(probability):
    """Generate personalized retention strategy based on churn probability."""
    if probability >= 80:
        return ("🚨 **High Risk:Offer a loyalty discount or personalized outreach to retain the customer. "
                "Consider direct intervention through phone calls, personalized emails, or exclusive renewal offers. "
                "Providing additional perks, such as early access to new features or limited-time discounts, can create urgency to stay. ")
    
    elif probability >= 60:
        return ("⚠️ Medium Risk: Engage with personalized promotions or enhanced customer service experience. "
                "Leverage targeted offers based on user history, such as special discounts on frequently used services. "
                "Send surveys or feedback requests to understand pain points and provide relevant solutions quickly.")
    
    elif probability >= 40:
        return ("🟡 Low Risk: Provide exclusive content, rewards, or perks to strengthen engagement. "
                "Introduce a points-based reward system or tiered loyalty benefits for continued usage. "
                "Offer insightful webinars, educational content, or premium support to encourage deeper product adoption. ")
    
    else:
        return ("✅ Minimal Risk: Maintain customer satisfaction with consistent value delivery. "
                "Focus on reinforcing positive experiences with high-quality service and reliable performance. "
                "Celebrate customer milestones with personalized appreciation messages or anniversary rewards. ")

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import textwrap

def generate_pdf():
    """Generate a comprehensive PDF report with corrected spacing, alignment, and automatic download."""
    
    customer_data = st.session_state.get('customer_profile', {})
    prediction, probability = st.session_state.get('prediction_result', ("N/A", 0))
    retention_strategy = get_retention_suggestion(probability * 100) if probability is not None else "Retention strategy unavailable."

    file_path = "customer_report.pdf"
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter
    y_position = height - 50

    # Title Section
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, y_position, "Customer Report")
    y_position -= 35

    # Customer Profile Section
    c.setFont("Helvetica", 12)
    c.drawString(100, y_position, "Customer Profile:")
    y_position -= 25

    # Customer Profile Table with Headers
    table_data = [["Feature", "Value"]] + [[key, str(value)] for key, value in customer_data.items()]
    table = Table(table_data, colWidths=[200, 200])

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Header background
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Header bold font
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    table.wrapOn(c, width, height)
    table.drawOn(c, 80, y_position - 200)
    y_position -= 230

    # Prediction Result Section
    c.drawString(100, y_position, "Prediction Result:")
    y_position -= 25
    c.drawString(100, y_position, f"Prediction: {prediction}")
    c.drawString(100, y_position - 20, f"Churn Probability: {probability * 100:.2f}%" if probability is not None else "N/A")
    y_position -= 50

    # Retention Strategy Section
    c.drawString(100, y_position, "Retention Strategy:")
    y_position -= 25

    wrapped_text = textwrap.wrap(retention_strategy, width=80)
    for line in wrapped_text:
        c.drawString(100, y_position, line)
        y_position -= 20

    y_position -= 50

    # Feature Importance Bar Chart Section
    c.drawString(100, y_position, "Feature Importance Chart:")
    y_position -= 25

    try:
        input_df = pd.DataFrame([customer_data])
        processed_df = pipeline.transform(input_df)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(processed_df)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feature_names = model.booster_.feature_name()
        importance_values = np.abs(shap_values).mean(axis=0)

        if len(importance_values) == len(feature_names):
            sorted_indices = np.argsort(importance_values)[::-1]
            feature_names = np.array(feature_names)[sorted_indices]
            importance_values = np.array(importance_values)[sorted_indices]

            plt.figure(figsize=(6, 3))  # Resized chart for PDF compatibility
            plt.barh(feature_names, importance_values, color="#008FEC", alpha=0.8)
            plt.xlabel("Feature Importance (SHAP)")
            plt.ylabel("Feature Name")
            plt.title("Impact of Features on Churn Prediction")
            plt.grid(axis="x", linestyle="--", alpha=0.5)
            plt.gca().invert_yaxis()

            image_path = "shap_feature_importance.png"
            plt.savefig(image_path, bbox_inches='tight', dpi=200)
            plt.close()

            if os.path.exists(image_path):
                c.drawImage(image_path, 100, y_position - 165, width=350, height=250)  # Adjusted positioning
        else:
            c.drawString(100, y_position, "Feature importance data does not match feature count.")
    
    except Exception as e:
        c.drawString(100, y_position, f"Error generating feature importance chart: {e}")

    c.save()
    return file_path

def report_page():   #Displays customer report and allows direct PDF download after prediction.
     
    st.markdown("""
    <div class="custom-subheader">Customer Profile:</div>
    """, unsafe_allow_html=True)

    customer_data = st.session_state.get('customer_profile', {})
    formatted_data = {key: customer_data.get(key, "N/A") for key in customer_data.keys()}   # Display Customer Profile
    customer_df = pd.DataFrame(list(formatted_data.items()), columns=["Feature", "Value"])
    st.table(customer_df)

    # Feature Importance Bar Chart
    st.markdown("""
    <div class="custom-subheader">Feature Importance Chart:</div>
    """, unsafe_allow_html=True)
    
    try:
        input_df = pd.DataFrame([customer_data])
        processed_df = pipeline.transform(input_df)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(processed_df)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feature_names = model.booster_.feature_name()
        importance_values = np.abs(shap_values).mean(axis=0)

        if len(importance_values) == len(feature_names):
            sorted_indices = np.argsort(importance_values)[::-1]
            feature_names = np.array(feature_names)[sorted_indices]
            importance_values = np.array(importance_values)[sorted_indices]

            plt.figure(figsize=(8, 5))
            plt.barh(feature_names, importance_values, color="#008FEC", alpha=0.8)
            plt.xlabel("Feature Importance (SHAP)")
            plt.ylabel("Feature Name")
            plt.title("Impact of Features on Churn Prediction")
            plt.grid(axis="x", linestyle="--", alpha=0.5)
            plt.gca().invert_yaxis()
            st.pyplot(plt)
        else:
            st.warning("Feature importance data does not match feature count.")

    except Exception :
        st.error(f"Error generating feature importance chart: Please run the prediction first !!")

    # Prediction Result
    st.markdown("""
    <div class="custom-subheader">Prediction Result:</div>
    """, unsafe_allow_html=True)
    
    prediction, probability = st.session_state.get('prediction_result', (None, None))
    if prediction is None or probability is None:
        st.warning("No prediction available. Please run the prediction first !!")
    else:
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Churn Probability:** {probability * 100:.2f}%")

    st.markdown("""
    <div class="custom-subheader">Retention Strategy:</div>
    """, unsafe_allow_html=True)
    
    if probability is not None:
        suggestion = get_retention_suggestion(probability * 100)
        st.write(suggestion)
    else:
        st.warning("Retention strategy unavailable due to missing prediction result ...")
        
    col1, col2, col3 = st.columns([10, 1, 2])  # Adjust column widths
    with col3:
        if "pdf_report" in st.session_state and st.session_state["pdf_report"]:
            with open(st.session_state["pdf_report"], "rb") as file:
                st.download_button(
                    label="⤓ Download ",
                    data=file,
                    file_name="customer_report.pdf",
                    mime="application/pdf"
                    )
        
# --- SHAP Analysis ---
def shap_analysis(customer_data):   #Displays SHAP feature importance and interpretability insights


    # Ensure customer data is properly initialized
    if not customer_data or customer_data is None:
        st.warning("Customer data is missing. Please run a prediction first.")
        return

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([customer_data])

    # Pass input through the pipeline for transformation
    try:
        processed_df = pipeline.transform(input_df)
        if processed_df is None:
            st.error("Pipeline transformation failed.")
            return
    except Exception as e:
        st.error(f"Error in pipeline transformation: {e}")
        return

    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed_df)

    # Select correct SHAP output if binary classification
    if isinstance(shap_values, list):  
        shap_values = shap_values[1]

    # Ensure correct feature importance mapping using actual feature names
    feature_names = model.booster_.feature_name()
    importance_values = np.abs(shap_values).mean(axis=0)

    if len(importance_values) == len(feature_names):
        sorted_indices = np.argsort(importance_values)[::-1]
        feature_names = np.array(feature_names)[sorted_indices]
        importance_values = np.array(importance_values)[sorted_indices]

        # **Store feature importance in session state**
        st.session_state["feature_importance"] = importance_values
        
        col1, col2 = st.columns([1.2,1.85])

        with col1:
            # SHAP Summary Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values, processed_df, feature_names=feature_names, show=False)
            plt.title("SHAP Summary Plot for Customer Churn", fontsize=16, color="#000000", pad=20)
            st.pyplot(fig)
        
        with col2:
            # Feature Importance Bar Chart
            plt.figure(figsize=(8, 5))
            plt.barh(feature_names, importance_values, color="#008FEC", alpha=0.8)
            plt.xlabel("Feature Importance (SHAP)")
            plt.ylabel("Feature Name")
            plt.title("Impact of Features on Churn Prediction")
            plt.grid(axis="x", linestyle="--", alpha=0.5)
            plt.gca().invert_yaxis()
            st.pyplot(plt)

        st.markdown("""
                    <div class="custom-subheader">SHAP Plot Explanation</div>
        """,unsafe_allow_html=True)
        
        # Business Interpretation of SHAP Analysis
        
        st.markdown("""     
        🔹 **Higher Importance Features:** Features at the top have a stronger impact on churn.\n
        🔹 **Positive vs. Negative Influence:** Red values increase churn risk, Blue values reduce it.\n
        🔹 **Business Strategy Insight:** Adjusting high-impact features can directly influence churn probability.
        """)
    else:
        st.warning("Feature importance data does not match feature count.")
        
def main():     # Main Function
    
    st.markdown('<div class="title">Churn Prediction Dashboard</div>', unsafe_allow_html=True)

    sidebar_churn_prediction()

    tab_names = ["Overview", "Model Performance", "Future Predictions", "SHAP Analysis", "Report"]
    tab_functions = [overview_page, performance_analysis_page, future_prediction_page, shap_analysis, report_page]

    tabs = st.tabs(tab_names)   #Tabs Setup

    for tab, func in zip(tabs, tab_functions):
        with tab:
            if func == shap_analysis:
                if 'customer_profile' in st.session_state:
                    customer_data = st.session_state['customer_profile']
                    func(customer_data)
                else:
                    st.warning("Customer data unavailable for SHAP analysis.")
            else:
                func()
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #7f8c8d; font-size: 14px;">
        <hr>
        <p>Customer Churn Prediction Model • Powered by LightGBM</p>
        <p>For business inquiries, please contact our Data Science Team</p>
    </div>
    """, unsafe_allow_html=True)
                
# Run App
if __name__ == "__main__":
    main()
