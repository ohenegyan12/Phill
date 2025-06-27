#!/usr/bin/env python3
"""
Student Performance Prediction - Web Application

A comprehensive Streamlit app with:
1. Interactive prediction interface
2. Data analysis and visualization
3. Model performance insights
4. Feature importance analysis
5. Model comparison tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_collection import DataCollector
from data_preprocessing import DataPreprocessor
from neural_network_model import StudentPerformanceModel

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .feature-input {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .grade-a { background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important; }
    .grade-b { background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%) !important; }
    .grade-c { background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%) !important; }
    .grade-d { background: linear-gradient(135deg, #fd7e14 0%, #e83e8c 100%) !important; }
    .grade-f { background: linear-gradient(135deg, #dc3545 0%, #6f42c1 100%) !important; }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: black;
    }
    
    .success-box {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    try:
        models_dir = 'models'
        if not os.path.exists(models_dir):
            return None, None
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        preprocessor_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        if not model_files or not preprocessor_files:
            return None, None
        
        latest_model = sorted(model_files)[-1]
        latest_preprocessor = sorted(preprocessor_files)[-1]
        
        model_path = os.path.join(models_dir, latest_model)
        preprocessor_path = os.path.join(models_dir, latest_preprocessor)
        
        model = StudentPerformanceModel(input_dim=50)
        model.load_model(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        return model, preprocessor
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        collector = DataCollector()
        df = collector.load_uci_dataset()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    """Main application function"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Student Performance Predictor</h1>
        <p>Predict student academic performance using advanced machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model, preprocessor = load_model_and_preprocessor()
    df = load_data()
    
    if model is None or preprocessor is None:
        st.error("⚠️ Model not found! Please run the training script first.")
        st.info("💡 Run: `python train_model.py` to train the model")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Home", "📈 Predict Performance", "📊 Data Insights", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "📈 Predict Performance":
        show_prediction_page(model, preprocessor)
    elif page == "📊 Data Insights":
        show_analysis_page(df)
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page(df):
    """Display the home page"""
    st.markdown("""
    ## 🎯 Welcome to the Student Performance Prediction System!
    
    This intelligent system helps predict student academic performance based on various factors that influence learning outcomes.
    
    ### ✨ What You Can Do:
    
    - **📈 Make Predictions**: Input student information and get accurate grade predictions
    - **📊 Explore Insights**: Understand what factors most influence academic success
    - **🎯 Get Recommendations**: Receive personalized insights for improvement
    
    ### 📋 Key Factors We Consider:
    
    | Category | Examples |
    |----------|----------|
    | **Academic History** | Previous grades, study time, attendance |
    | **Personal Factors** | Age, health, motivation |
    | **Family Background** | Parental education, family support |
    | **Lifestyle** | Free time activities, social life |
    | **Support Systems** | School support, extra classes |
    
    ### 🎯 Prediction Target:
    - **Final Grade (G3)**: Predicted final grade on a 0-20 scale
    """)
    
    if df is not None:
        # Quick stats with better styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Total Students", len(df), help="Number of students in our dataset")
        with col2:
            st.metric("📊 Average Grade", f"{df['G3'].mean():.1f}", help="Average final grade across all students")
        with col3:
            st.metric("🏆 Highest Grade", f"{df['G3'].max():.1f}", help="Best performing student's grade")
        with col4:
            st.metric("📈 Lowest Grade", f"{df['G3'].min():.1f}", help="Lowest performing student's grade")
        
        # Quick visualization
        st.markdown("### 📊 Grade Distribution")
        fig = px.histogram(df, x='G3', nbins=20, 
                          title="Distribution of Final Grades",
                          labels={'G3': 'Final Grade', 'count': 'Number of Students'},
                          color_discrete_sequence=['#667eea'])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(model, preprocessor):
    """Display the prediction page (updated UI)"""
    st.markdown("## 📈 Student Performance Prediction")
    st.markdown("""
    <div class="info-box">
        💡 <strong>How to use:</strong> Fill in the student information below and click "Predict" to get an accurate prediction of their exam score.
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("### Enter Student Data to Predict Performance")
        hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=100.0, value=5.0, help="Enter the number of hours the student studied for the exam.")
        previous_score = st.number_input("Previous Test Score", min_value=0.0, max_value=100.0, value=75.0, help="Enter the score from the student's previous test (0-100).")
        attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=90.0, help="Enter the attendance rate in percentage (0-100%).")
        extracurricular = st.selectbox("Extra-Curricular Participation (Yes=1, No=0)", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No", help="Enter 1 if the student participates in extra-curricular activities, 0 otherwise.")
        parent_edu = st.selectbox("Parent's Education Level (1=High School, 2=Undergraduate, 3=Graduate)", options=[1, 2, 3], format_func=lambda x: {1: 'High School', 2: 'Undergraduate', 3: 'Graduate'}[x], help="Enter the highest education level of the student's parents: 1 for High School, 2 for Undergraduate, and 3 for Graduate.")
        submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                input_data = {
                    'Hours_Studied': hours_studied,
                    'Previous_Scores': previous_score,
                    'Attendance': attendance,
                    'Extracurricular_Activities': extracurricular,
                    'Parental_Education_Level': parent_edu
                }
                input_df = pd.DataFrame([input_data])
                X_transformed = preprocessor.transform_new_data(input_df)
                prediction = model.predict(X_transformed)[0][0]
                st.success(f"Predicted Exam Score: {prediction:.1f}")
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")

def show_analysis_page(df):
    """Display the analysis page"""
    st.markdown("## 📊 Data Insights & Analysis")
    
    if df is None:
        st.error("❌ Data not available")
        return
    
    # Overview with better styling
    st.markdown("### 📈 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📚 Total Students", len(df))
        st.metric("📊 Features", len(df.columns))
    with col2:
        st.metric("📈 Average Final Grade", f"{df['G3'].mean():.1f}")
        st.metric("📊 Grade Std Dev", f"{df['G3'].std():.1f}")
    with col3:
        st.metric("👨 Male Students", f"{len(df[df['sex'] == 'M'])} ({len(df[df['sex'] == 'M'])/len(df)*100:.1f}%)")
        st.metric("👩 Female Students", f"{len(df[df['sex'] == 'F'])} ({len(df[df['sex'] == 'F'])/len(df)*100:.1f}%)")
    
    # Grade distribution
    st.markdown("### 📊 Grade Distribution")
    fig = px.histogram(df, x='G3', nbins=20, 
                      title="Distribution of Final Grades",
                      labels={'G3': 'Final Grade', 'count': 'Number of Students'},
                      color_discrete_sequence=['#667eea'])
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### 🔗 Feature Correlations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['G3'].sort_values(ascending=False)
    correlations = correlations[correlations.index != 'G3']
    
    fig = px.bar(x=correlations.values, y=correlations.index, 
                 orientation='h',
                 title="Feature Correlations with Final Grade (G3)",
                 labels={'x': 'Correlation Coefficient', 'y': 'Features'},
                 color=correlations.values,
                 color_continuous_scale='RdBu')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis
    st.markdown("### 📋 Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📚 Academic Factors", "👥 Demographics", "🏃 Lifestyle"])
    
    with tab1:
        st.markdown("#### Academic Performance Factors")
        
        # G1 vs G3
        fig = px.scatter(df, x='G1', y='G3', color='sex',
                        title="First Period Grade vs Final Grade",
                        labels={'G1': 'First Period Grade', 'G3': 'Final Grade'},
                        color_discrete_map={'M': '#667eea', 'F': '#764ba2'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Study time analysis
        study_time_stats = df.groupby('studytime')['G3'].agg(['mean', 'count']).reset_index()
        fig = px.bar(study_time_stats, x='studytime', y='mean',
                    title="Average Final Grade by Study Time",
                    labels={'studytime': 'Study Time', 'mean': 'Average Final Grade'},
                    color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Demographic Analysis")
        
        # Gender analysis
        gender_stats = df.groupby('sex')['G3'].agg(['mean', 'count']).reset_index()
        fig = px.bar(gender_stats, x='sex', y='mean',
                    title="Average Final Grade by Gender",
                    labels={'sex': 'Gender', 'mean': 'Average Final Grade'},
                    color_discrete_sequence=['#667eea', '#764ba2'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Age analysis
        age_stats = df.groupby('age')['G3'].agg(['mean', 'count']).reset_index()
        fig = px.bar(age_stats, x='age', y='mean',
                    title="Average Final Grade by Age",
                    labels={'age': 'Age', 'mean': 'Average Final Grade'},
                    color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Lifestyle Factors")
        
        # Health vs Grade
        health_stats = df.groupby('health')['G3'].agg(['mean', 'count']).reset_index()
        fig = px.bar(health_stats, x='health', y='mean',
                    title="Average Final Grade by Health Status",
                    labels={'health': 'Health Status', 'mean': 'Average Final Grade'},
                    color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Absences vs Grade
        fig = px.scatter(df, x='absences', y='G3',
                        title="Absences vs Final Grade",
                        labels={'absences': 'Number of Absences', 'G3': 'Final Grade'},
                        color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    """Display the about page"""
    st.markdown("## ℹ️ About This System")
    
    st.markdown("""
    ### 🎓 Student Performance Prediction System
    
    This intelligent system helps predict student academic performance using advanced machine learning techniques.
    
    ### 🎯 What We Do
    
    We analyze various factors that influence student success and provide accurate predictions to help:
    - **Students** understand their potential performance
    - **Educators** identify students who might need additional support
    - **Parents** understand factors affecting their child's academic success
    
    ### 📊 What We Consider
    
    Our system analyzes multiple factors including:
    
    #### Academic Factors
    - Previous grades and performance history
    - Study time and learning habits
    - Attendance and participation
    
    #### Personal Factors
    - Age and maturity level
    - Health and well-being
    - Motivation and engagement
    
    #### Family Background
    - Parental education levels
    - Family support and relationships
    - Socioeconomic factors
    
    #### Lifestyle Factors
    - Free time activities
    - Social interactions
    - Health habits
    
    ### 🎯 Our Predictions
    
    - **Target**: Final Grade (G3) on a 0-20 scale
    - **Accuracy**: High precision predictions based on comprehensive data analysis
    - **Interpretation**: Clear grade levels (A-F) with actionable insights
    
    ### 🔒 Privacy & Ethics
    
    - This system uses synthetic data for demonstration purposes
    - No real student information is collected or stored
    - Educational AI should be used responsibly and transparently
    - Consider potential biases and fairness implications
    
    ### 💡 How to Use
    
    1. Navigate to "Predict Performance"
    2. Input student information using the form
    3. Click "Predict" to get results
    4. Review the prediction and recommendations
    5. Explore data insights for deeper understanding
    
    ---
    
    **Built with ❤️ for educational advancement**
    
    *Empowering students, educators, and parents with data-driven insights*
    """)

if __name__ == "__main__":
    main() 