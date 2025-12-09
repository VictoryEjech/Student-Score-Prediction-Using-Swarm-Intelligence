# advanced_student_analytics_system.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
import base64
from io import BytesIO
import uuid
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Advanced Student Analytics System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        padding: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sub-header styling */
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #E5E7EB;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.12);
    }
    
    .success-card {
        border-left: 6px solid #10B981;
        background: linear-gradient(135deg, #D1FAE5 0%, #ECFDF5 100%);
    }
    
    .warning-card {
        border-left: 6px solid #F59E0B;
        background: linear-gradient(135deg, #FEF3C7 0%, #FFFBEB 100%);
    }
    
    .danger-card {
        border-left: 6px solid #EF4444;
        background: linear-gradient(135deg, #FEE2E2 0%, #FEF2F2 100%);
    }
    
    .info-card {
        border-left: 6px solid #3B82F6;
        background: linear-gradient(135deg, #DBEAFE 0%, #EFF6FF 100%);
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        padding: 14px 28px;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(59, 130, 246, 0.4);
        background: linear-gradient(90deg, #2563EB 0%, #1E40AF 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #F3F4F6;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Feature link styling */
    .feature-link {
        display: inline-block;
        background: linear-gradient(90deg, #8B5CF6 0%, #7C3AED 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        text-decoration: none;
        margin: 5px;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }
    
    .feature-link:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(139, 92, 246, 0.3);
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #E0F2FE 0%, #F0F9FF 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #BAE6FD;
        margin: 15px 0;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10B981 0%, #34D399 100%);
    }
    
    /* Navigation buttons */
    .nav-button {
        display: inline-block;
        padding: 10px 20px;
        margin: 5px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .nav-button-primary {
        background: linear-gradient(90deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
    }
    
    .nav-button-secondary {
        background: #F3F4F6;
        color: #4B5563;
        border: 2px solid #D1D5DB;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #1F2937;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedStudentAnalyticsSystem:
    def __init__(self):
        """Initialize the system"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.batch_results = None
        self.comparison_data = None
        self.initialize_session_state()
        self.load_models()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'batch_data' not in st.session_state:
            st.session_state.batch_data = None
        if 'batch_predictions' not in st.session_state:
            st.session_state.batch_predictions = None
        if 'single_prediction' not in st.session_state:
            st.session_state.single_prediction = None
        if 'comparison_mode' not in st.session_state:
            st.session_state.comparison_mode = False
    
    def load_models(self):
        """Load trained model and scaler"""
        try:
            # Try to load the model
            self.model = joblib.load('student_score_predictor.pkl')
            self.scaler = joblib.load('feature_scaler.pkl')
            
            # Define expected features
            self.feature_names = [
                'previous_gpa', 'attendance_rate', 'assignment_completion',
                'study_hours_weekly', 'library_visits', 'online_activity',
                'gender', 'age', 'midterm_score', 'quiz_average',
                'sleep_hours', 'stress_level', 'gpa_attendance_interaction',
                'study_completion_interaction', 'sleep_study_interaction'
            ]
            
            st.sidebar.success("✅ **Models Loaded Successfully!**")
            
        except FileNotFoundError:
            # Create a simulated model for demo purposes
            st.sidebar.warning("⚠️ **Model files not found. Using simulated model for demonstration.**")
            st.sidebar.info("To use the real model, run `train_model.py` first to generate model files.")
            self.create_simulated_model()
    
    def create_simulated_model(self):
        """Create a simulated model for demonstration"""
        class SimulatedModel:
            def __init__(self):
                self.feature_importances_ = np.random.rand(15)
            
            def predict(self, X):
                # Simulate predictions based on features
                predictions = []
                for features in X:
                    score = (
                        0.3 * (features[0] - 2.0) / 2.0 +  # GPA
                        0.2 * (features[1] - 0.5) / 0.5 +  # Attendance
                        0.15 * (features[3] - 10) / 20 +   # Study hours
                        np.random.normal(0, 0.1)
                    )
                    predictions.append(1 if score > 0 else 0)
                return np.array(predictions)
            
            def predict_proba(self, X):
                # Simulate probabilities
                proba = []
                for features in X:
                    base_prob = (
                        0.3 * (features[0] - 2.0) / 2.0 +
                        0.2 * (features[1] - 0.5) / 0.5 +
                        0.15 * (features[3] - 10) / 20
                    )
                    prob_high = 1 / (1 + np.exp(-base_prob))
                    prob_high = np.clip(prob_high, 0.05, 0.95)
                    proba.append([1 - prob_high, prob_high])
                return np.array(proba)
        
        class SimulatedScaler:
            def transform(self, X):
                return X  # No scaling in simulation
            def fit(self, X):
                return self
        
        self.model = SimulatedModel()
        self.scaler = SimulatedScaler()
        self.feature_names = [
            'previous_gpa', 'attendance_rate', 'assignment_completion',
            'study_hours_weekly', 'library_visits', 'online_activity',
            'gender', 'age', 'midterm_score', 'quiz_average',
            'sleep_hours', 'stress_level', 'gpa_attendance_interaction',
            'study_completion_interaction', 'sleep_study_interaction'
        ]
    
    def create_single_student_interface(self):
        """Create interface for single student prediction"""
        st.markdown('<div class="sub-header">🎯 Single Student Prediction</div>', unsafe_allow_html=True)
        
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                <div class="info-box">
                <h4>📝 Enter individual student details for personalized prediction and recommendations.</h4>
                <p>Fill in all the fields below to get an accurate prediction of the student's performance.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #E0F2FE 0%, #F0F9FF 100%); border-radius: 10px;">
                <h4>⚡ Quick Tips</h4>
                <p>• Complete all fields for best accuracy</p>
                <p>• Use realistic values based on student history</p>
                <p>• Review recommendations carefully</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Create input form with tabs for different categories
        form_tabs = st.tabs(["📊 Academic Info", "⏰ Study Habits", "😴 Lifestyle", "📈 Current Performance"])
        
        with form_tabs[0]:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                student_name = st.text_input("Student Name", placeholder="Enter student name", key="name")
                student_id = st.text_input("Student ID", placeholder="STU-001", key="id")
                age = st.number_input("Age", min_value=16, max_value=40, value=21, step=1, key="age")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
                major = st.selectbox("Major/Program", [
                    "Computer Science", "Engineering", "Business", "Biology", 
                    "Psychology", "Mathematics", "Physics", "Chemistry", 
                    "Economics", "Literature", "Other"
                ], key="major")
            
            with col2:
                previous_gpa = st.slider(
                    "Previous Semester GPA",
                    min_value=1.0, max_value=4.0, value=3.2, step=0.1,
                    help="Cumulative GPA from previous semester",
                    key="gpa"
                )
                
                attendance_rate = st.slider(
                    "Class Attendance Rate",
                    min_value=0.0, max_value=1.0, value=0.85, step=0.01,
                    format="%.0f%%",
                    help="Percentage of classes attended",
                    key="attendance"
                )
            
            with col3:
                assignment_completion = st.slider(
                    "Assignment Completion Rate",
                    min_value=0.0, max_value=1.0, value=0.88, step=0.01,
                    format="%.0f%%",
                    help="Percentage of assignments completed",
                    key="assignments"
                )
                
                online_activity = st.slider(
                    "Online Learning Activity",
                    min_value=0.0, max_value=1.0, value=0.75, step=0.01,
                    format="Low - High",
                    help="Level of engagement with online materials",
                    key="online"
                )
        
        with form_tabs[1]:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                study_hours_weekly = st.slider(
                    "Weekly Study Hours",
                    min_value=0, max_value=40, value=15, step=1,
                    help="Average hours spent studying per week",
                    key="study_hours"
                )
            
            with col2:
                library_visits = st.slider(
                    "Library Visits (per month)",
                    min_value=0, max_value=20, value=4, step=1,
                    help="Number of times visiting library each month",
                    key="library"
                )
            
            with col3:
                study_group = st.selectbox(
                    "Study Group Participation",
                    ["Never", "Occasionally", "Weekly", "Daily"],
                    key="study_group"
                )
        
        with form_tabs[2]:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sleep_hours = st.slider(
                    "Sleep Hours (per night)",
                    min_value=4.0, max_value=10.0, value=7.0, step=0.5,
                    help="Average hours of sleep per night",
                    key="sleep"
                )
            
            with col2:
                stress_level = st.slider(
                    "Stress Level",
                    min_value=1, max_value=10, value=5, step=1,
                    help="Self-reported stress level (1=Low, 10=High)",
                    key="stress"
                )
            
            with col3:
                has_job = st.checkbox("Has Part-time Job", key="has_job")
                job_hours = 0
                if has_job:
                    job_hours = st.slider(
                        "Job Hours per Week",
                        min_value=5, max_value=40, value=15, step=1,
                        key="job_hours"
                    )
        
        with form_tabs[3]:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                midterm_score = st.slider(
                    "Midterm Exam Score",
                    min_value=30, max_value=100, value=72, step=1,
                    format="%d",
                    help="Score on midterm examination",
                    key="midterm"
                )
            
            with col2:
                quiz_average = st.slider(
                    "Quiz Average Score",
                    min_value=40, max_value=100, value=75, step=1,
                    format="%d",
                    help="Average score on all quizzes",
                    key="quiz"
                )
            
            with col3:
                project_score = st.slider(
                    "Project Score (if applicable)",
                    min_value=0, max_value=100, value=80, step=1,
                    format="%d",
                    help="Score on major projects",
                    key="project"
                )
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(
                "🚀 Generate Prediction & Recommendations",
                use_container_width=True,
                type="primary",
                key="predict_single"
            )
        
        # Store input data
        input_data = {
            'student_name': student_name,
            'student_id': student_id,
            'age': age,
            'gender': 1 if gender == "Female" else (0 if gender == "Male" else 2),
            'major': major,
            'previous_gpa': previous_gpa,
            'attendance_rate': attendance_rate,
            'assignment_completion': assignment_completion,
            'online_activity': online_activity,
            'study_hours_weekly': study_hours_weekly,
            'library_visits': library_visits,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'job_hours': job_hours,
            'midterm_score': midterm_score,
            'quiz_average': quiz_average,
            'project_score': project_score
        }
        
        return predict_button, input_data
    
    def process_single_prediction(self, input_data):
        """Process single student prediction"""
        # Prepare features
        features = self.prepare_features(input_data)
        
        # Make prediction
        prediction, probability = self.predict_single(features)
        
        # Store in session state
        st.session_state.single_prediction = {
            'prediction': prediction,
            'probability': probability,
            'features': features,
            'input_data': input_data,
            'timestamp': datetime.now()
        }
        
        return prediction, probability, features
    
    def prepare_features(self, input_data):
        """Prepare features for prediction"""
        features_dict = {}
        
        # Extract and calculate features
        features_dict['previous_gpa'] = input_data['previous_gpa']
        features_dict['attendance_rate'] = input_data['attendance_rate']
        features_dict['assignment_completion'] = input_data['assignment_completion']
        features_dict['study_hours_weekly'] = input_data['study_hours_weekly']
        features_dict['library_visits'] = input_data['library_visits']
        features_dict['online_activity'] = input_data['online_activity']
        features_dict['gender'] = input_data['gender']
        features_dict['age'] = input_data['age']
        features_dict['midterm_score'] = input_data['midterm_score']
        features_dict['quiz_average'] = input_data['quiz_average']
        features_dict['sleep_hours'] = input_data['sleep_hours']
        features_dict['stress_level'] = input_data['stress_level']
        
        # Calculate interaction features
        features_dict['gpa_attendance_interaction'] = (
            input_data['previous_gpa'] * input_data['attendance_rate']
        )
        features_dict['study_completion_interaction'] = (
            input_data['study_hours_weekly'] * input_data['assignment_completion']
        )
        features_dict['sleep_study_interaction'] = (
            input_data['sleep_hours'] * input_data['study_hours_weekly'] / 10
        )
        
        # Create DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Reorder columns
        features_df = features_df[self.feature_names]
        
        return features_df
    
    def predict_single(self, features_df):
        """Make prediction for single student"""
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probability
    
    def display_single_prediction_results(self, prediction, probability, features, input_data):
        """Display single prediction results"""
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 Prediction Results</div>', unsafe_allow_html=True)
        
        # Result cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            card_class = "success-card" if prediction == 1 else "danger-card"
            st.markdown(f'<div class="card {card_class}">', unsafe_allow_html=True)
            st.metric(
                label="Predicted Outcome",
                value="🏆 HIGH SCORE" if prediction == 1 else "⚠️ AT RISK",
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            confidence = probability[1] if prediction == 1 else probability[0]
            st.markdown('<div class="card info-card">', unsafe_allow_html=True)
            st.metric(
                label="Confidence Level",
                value=f"{confidence*100:.1f}%",
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            risk_score = self.calculate_risk_score(input_data)
            st.markdown('<div class="card warning-card">', unsafe_allow_html=True)
            st.metric(
                label="Risk Score",
                value=f"{risk_score:.0f}/100",
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="card info-card">', unsafe_allow_html=True)
            recommendation = "MAINTAIN" if prediction == 1 else "INTERVENE"
            st.metric(
                label="Recommended Action",
                value=recommendation,
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed analysis
        st.markdown("### 📋 Detailed Analysis")
        
        analysis_tabs = st.tabs(["🎯 Recommendations", "📈 Performance Metrics", "📊 Feature Impact"])
        
        with analysis_tabs[0]:
            self.display_recommendations(prediction, input_data)
        
        with analysis_tabs[1]:
            self.display_performance_metrics(input_data)
        
        with analysis_tabs[2]:
            self.display_feature_impact(features)
        
        # Export options
        st.markdown("### 💾 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Report", use_container_width=True):
                self.download_single_report(input_data, prediction, probability)
        
        with col2:
            if st.button("🔄 New Prediction", use_container_width=True):
                st.rerun()
        
        with col3:
            if st.button("➕ Add to Batch", use_container_width=True):
                self.add_to_batch(input_data, prediction, probability)
    
    def calculate_risk_score(self, input_data):
        """Calculate comprehensive risk score"""
        risk_factors = []
        
        # Academic risks
        if input_data['previous_gpa'] < 2.5:
            risk_factors.append(30)
        elif input_data['previous_gpa'] < 3.0:
            risk_factors.append(15)
        
        if input_data['attendance_rate'] < 0.7:
            risk_factors.append(25)
        elif input_data['attendance_rate'] < 0.8:
            risk_factors.append(12)
        
        if input_data['assignment_completion'] < 0.8:
            risk_factors.append(20)
        
        # Behavioral risks
        if input_data['study_hours_weekly'] < 10:
            risk_factors.append(15)
        
        if input_data['stress_level'] > 7:
            risk_factors.append(20)
        
        if input_data['sleep_hours'] < 6:
            risk_factors.append(15)
        
        if input_data.get('job_hours', 0) > 20:
            risk_factors.append(10)
        
        # Calculate total risk
        risk_score = min(100, sum(risk_factors))
        return risk_score
    
    def display_recommendations(self, prediction, input_data):
        """Display personalized recommendations"""
        recommendations = []
        
        if prediction == 1:  # High score predicted
            st.success("🎉 **Excellent! The student is predicted to achieve high scores.**")
            
            recommendations.append("✅ **Continue current study habits** - Your approach is working well")
            
            if input_data['attendance_rate'] > 0.9:
                recommendations.append("📅 **Attendance is excellent** - Maintain this level")
            else:
                recommendations.append("📅 **Aim for 90%+ attendance** - Small improvements can help")
            
            if input_data['study_hours_weekly'] < 15:
                recommendations.append("📚 **Consider increasing study time to 15-20 hours/week**")
            else:
                recommendations.append("⏰ **Study schedule is well-balanced**")
            
            recommendations.append("🌟 **Explore advanced topics or peer tutoring opportunities**")
            
        else:  # At-risk prediction
            st.warning("⚠️ **Attention needed: Student is at risk of underperforming.**")
            
            recommendations.append("🔍 **Immediate intervention recommended**")
            
            if input_data['attendance_rate'] < 0.8:
                recommendations.append(f"📅 **Increase attendance** from {input_data['attendance_rate']*100:.0f}% to at least 80%")
            
            if input_data['assignment_completion'] < 0.9:
                recommendations.append(f"📝 **Complete all assignments** (currently {input_data['assignment_completion']*100:.0f}%)")
            
            if input_data['study_hours_weekly'] < 10:
                recommendations.append(f"⏰ **Study more hours** (currently {input_data['study_hours_weekly']:.1f} hrs/week)")
            
            if input_data['sleep_hours'] < 7:
                recommendations.append(f"😴 **Get more sleep** (currently {input_data['sleep_hours']:.1f} hrs/night)")
            
            if input_data['stress_level'] > 6:
                recommendations.append(f"🧘 **Manage stress** (current level: {input_data['stress_level']:.1f}/10)")
            
            recommendations.append("👥 **Join a study group or seek tutoring**")
            recommendations.append("💡 **Meet with instructor during office hours**")
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    
    def display_performance_metrics(self, input_data):
        """Display performance metrics"""
        # Create gauge charts
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=('Academic Performance', 'Study Habits', 
                          'Lifestyle Balance', 'Current Grades')
        )
        
        # Academic Performance Score
        academic_score = (
            (input_data['previous_gpa'] / 4.0 * 40) +
            (input_data['attendance_rate'] * 30) +
            (input_data['assignment_completion'] * 30)
        )
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=academic_score,
            title={'text': "Score"},
            domain={'row': 0, 'column': 0},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#3B82F6"},
                   'steps': [
                       {'range': [0, 60], 'color': "#EF4444"},
                       {'range': [60, 80], 'color': "#F59E0B"},
                       {'range': [80, 100], 'color': "#10B981"}
                   ]}
        ), row=1, col=1)
        
        # Study Habits Score
        study_score = (
            (input_data['study_hours_weekly'] / 30 * 40) +
            (input_data['online_activity'] * 30) +
            (min(input_data['library_visits'], 10) / 10 * 30)
        )
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=study_score,
            title={'text': "Score"},
            domain={'row': 0, 'column': 1},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#8B5CF6"},
                   'steps': [
                       {'range': [0, 60], 'color': "#EF4444"},
                       {'range': [60, 80], 'color': "#F59E0B"},
                       {'range': [80, 100], 'color': "#10B981"}
                   ]}
        ), row=1, col=2)
        
        # Lifestyle Score
        lifestyle_score = (
            (input_data['sleep_hours'] / 10 * 40) +
            ((10 - input_data['stress_level']) / 10 * 30) +
            (max(0, 30 - input_data.get('job_hours', 0) * 0.5))
        )
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=lifestyle_score,
            title={'text': "Score"},
            domain={'row': 1, 'column': 0},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#10B981"},
                   'steps': [
                       {'range': [0, 60], 'color': "#EF4444"},
                       {'range': [60, 80], 'color': "#F59E0B"},
                       {'range': [80, 100], 'color': "#10B981"}
                   ]}
        ), row=2, col=1)
        
        # Current Grades Score
        grades_score = (input_data['midterm_score'] + input_data['quiz_average']) / 2
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=grades_score,
            title={'text': "Score"},
            domain={'row': 1, 'column': 1},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#F59E0B"},
                   'steps': [
                       {'range': [0, 60], 'color': "#EF4444"},
                       {'range': [60, 80], 'color': "#F59E0B"},
                       {'range': [80, 100], 'color': "#10B981"}
                   ]}
        ), row=2, col=2)
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_feature_impact(self, features):
        """Display feature importance and impact"""
        if hasattr(self.model, 'feature_importances_'):
            # Create feature importance visualization
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df.tail(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Most Important Features',
                color='Importance',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show current student's feature values
            st.markdown("### 📝 Current Student's Feature Values")
            feature_values = pd.DataFrame({
                'Feature': self.feature_names,
                'Value': features.iloc[0].values
            }).sort_values('Feature')
            
            st.dataframe(feature_values, use_container_width=True)
        else:
            st.info("Feature importance data is not available for the current model.")
    
    def download_single_report(self, input_data, prediction, probability):
        """Generate and download single student report"""
        report = f"""
        STUDENT PERFORMANCE PREDICTION REPORT
        =====================================
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        STUDENT INFORMATION:
        - Name: {input_data.get('student_name', 'N/A')}
        - ID: {input_data.get('student_id', 'N/A')}
        - Age: {input_data.get('age', 'N/A')}
        - Major: {input_data.get('major', 'N/A')}
        
        ACADEMIC METRICS:
        - Previous GPA: {input_data.get('previous_gpa', 'N/A')}
        - Attendance Rate: {input_data.get('attendance_rate', 'N/A')*100:.1f}%
        - Assignment Completion: {input_data.get('assignment_completion', 'N/A')*100:.1f}%
        - Study Hours Weekly: {input_data.get('study_hours_weekly', 'N/A')}
        - Midterm Score: {input_data.get('midterm_score', 'N/A')}
        - Quiz Average: {input_data.get('quiz_average', 'N/A')}
        
        PREDICTION RESULTS:
        - Predicted Outcome: {'HIGH SCORE' if prediction == 1 else 'AT RISK'}
        - Probability: {probability[1]*100:.1f}% (High Score)
        - Confidence Level: {'High' if max(probability) > 0.8 else 'Medium' if max(probability) > 0.6 else 'Low'}
        - Risk Score: {self.calculate_risk_score(input_data):.0f}/100
        
        RECOMMENDATIONS:
        """
        
        # Add recommendations based on prediction
        if prediction == 1:
            report += "\n- Continue current study habits"
            report += "\n- Maintain high attendance levels"
            report += "\n- Consider advanced learning opportunities"
        else:
            report += "\n- Increase class attendance"
            report += "\n- Complete all assignments on time"
            report += "\n- Increase weekly study hours"
            report += "\n- Seek academic support if needed"
        
        # Convert to downloadable format
        b64 = base64.b64encode(report.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="student_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt">Click to download report</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def add_to_batch(self, input_data, prediction, probability):
        """Add single prediction to batch data"""
        if 'batch_data' not in st.session_state:
            st.session_state.batch_data = []
        
        st.session_state.batch_data.append({
            **input_data,
            'prediction': prediction,
            'probability_high': probability[1],
            'probability_low': probability[0],
            'risk_score': self.calculate_risk_score(input_data),
            'timestamp': datetime.now()
        })
        
        st.success(f"✅ Student added to batch. Total students in batch: {len(st.session_state.batch_data)}")
    
    def create_batch_prediction_interface(self):
        """Create interface for batch prediction"""
        st.markdown('<div class="sub-header">📊 Batch Student Prediction</div>', unsafe_allow_html=True)
        
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                <div class="info-box">
                <h4>📁 Upload or enter data for multiple students to analyze performance at scale.</h4>
                <p>Choose your preferred input method and follow the instructions below.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #D1FAE5 0%, #ECFDF5 100%); border-radius: 10px;">
                <h4>📈 Batch Benefits</h4>
                <p>• Analyze 100+ students simultaneously</p>
                <p>• Identify group trends and patterns</p>
                <p>• Generate comprehensive reports</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Input method selection
        st.markdown("### 📥 Select Input Method")
        
        input_tabs = st.tabs(["📁 Upload File", "📝 Manual Entry", "🎯 Sample Data"])
        
        batch_df = pd.DataFrame()
        
        with input_tabs[0]:
            st.markdown("""
            <div class="info-card" style="padding: 15px; margin-bottom: 20px;">
            <h5>📋 File Format Requirements:</h5>
            <p>• CSV or Excel format</p>
            <p>• Include required columns: previous_gpa, attendance_rate, assignment_completion, study_hours_weekly</p>
            <p>• Optional columns: midterm_score, quiz_average, sleep_hours, stress_level</p>
            <p>• <a href="#" onclick="alert('Download template from: templates/student_data_template.csv')">Download template</a></p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV or Excel file with student data"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        batch_df = pd.read_csv(uploaded_file)
                    else:
                        batch_df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ Successfully loaded {len(batch_df)} student records")
                    st.dataframe(batch_df.head(), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        
        with input_tabs[1]:
            st.markdown("""
            <div class="info-card" style="padding: 15px; margin-bottom: 20px;">
            <h5>📝 Manual Data Entry:</h5>
            <p>• Enter data for each student in the table below</p>
            <p>• Click 'Add Row' to add more students</p>
            <p>• Minimum required fields: Student ID, GPA, Attendance</p>
            </div>
            """, unsafe_allow_html=True)
            
            num_students = st.number_input(
                "Number of students to enter:",
                min_value=1,
                max_value=100,
                value=5,
                step=1
            )
            
            if num_students > 0:
                # Create template data
                template_data = []
                for i in range(num_students):
                    template_data.append({
                        'student_id': f'STU{1000 + i}',
                        'name': f'Student {i+1}',
                        'previous_gpa': 3.0,
                        'attendance_rate': 0.8,
                        'assignment_completion': 0.85,
                        'study_hours_weekly': 15,
                        'midterm_score': 75,
                        'quiz_average': 72,
                        'sleep_hours': 7.0,
                        'stress_level': 5
                    })
                
                template_df = pd.DataFrame(template_data)
                
                # Editable dataframe
                edited_df = st.data_editor(
                    template_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "student_id": st.column_config.TextColumn("Student ID", required=True),
                        "name": st.column_config.TextColumn("Student Name"),
                        "previous_gpa": st.column_config.NumberColumn(
                            "Previous GPA",
                            min_value=1.0,
                            max_value=4.0,
                            step=0.1
                        ),
                        "attendance_rate": st.column_config.NumberColumn(
                            "Attendance Rate",
                            min_value=0.0,
                            max_value=1.0,
                            step=0.01,
                            format="%.2f"
                        ),
                        "assignment_completion": st.column_config.NumberColumn(
                            "Assignment Completion",
                            min_value=0.0,
                            max_value=1.0,
                            step=0.01,
                            format="%.2f"
                        ),
                        "study_hours_weekly": st.column_config.NumberColumn(
                            "Study Hours Weekly",
                            min_value=0,
                            max_value=40,
                            step=1
                        ),
                        "midterm_score": st.column_config.NumberColumn(
                            "Midterm Score",
                            min_value=0,
                            max_value=100,
                            step=1
                        ),
                        "quiz_average": st.column_config.NumberColumn(
                            "Quiz Average",
                            min_value=0,
                            max_value=100,
                            step=1
                        ),
                        "sleep_hours": st.column_config.NumberColumn(
                            "Sleep Hours",
                            min_value=4.0,
                            max_value=10.0,
                            step=0.5
                        ),
                        "stress_level": st.column_config.NumberColumn(
                            "Stress Level",
                            min_value=1,
                            max_value=10,
                            step=1
                        )
                    }
                )
                
                batch_df = edited_df.copy()
        
        with input_tabs[2]:
            st.markdown("""
            <div class="info-card" style="padding: 15px; margin-bottom: 20px;">
            <h5>🎯 Sample Data Generation:</h5>
            <p>• Generate realistic sample data for testing</p>
            <p>• Adjust the sample size as needed</p>
            <p>• Data includes realistic patterns and correlations</p>
            </div>
            """, unsafe_allow_html=True)
            
            sample_size = st.slider("Sample size:", 5, 100, 20)
            
            if st.button("🎲 Generate Sample Data", use_container_width=True):
                batch_df = self.generate_sample_data(sample_size)
                st.success(f"✅ Generated {sample_size} sample student records")
                st.dataframe(batch_df.head(), use_container_width=True)
        
        # Process batch data if available
        if not batch_df.empty:
            # Add missing features with default values
            batch_df = self.add_missing_features(batch_df)
            
            # Store in session state
            st.session_state.batch_data = batch_df
            
            # Run prediction button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                    with st.spinner("🔮 Analyzing student batch..."):
                        # Run batch prediction
                        predictions_df = self.predict_batch(batch_df)
                        
                        # Store results
                        st.session_state.batch_predictions = predictions_df
                        
                        # Enable comparative analysis
                        st.session_state.comparison_mode = True
                        
                        st.success(f"✅ Batch prediction complete! {len(predictions_df)} students analyzed.")
                        
                        # Show summary
                        self.display_batch_summary(predictions_df)
        
        # Show existing batch data if available
        if st.session_state.batch_data is not None and not st.session_state.batch_data.empty:
            st.markdown("### 📋 Current Batch Data")
            st.dataframe(st.session_state.batch_data.head(), use_container_width=True)
            
            if st.session_state.batch_predictions is not None:
                st.markdown("### 📊 Batch Predictions")
                self.display_batch_predictions(st.session_state.batch_predictions)
    
    def generate_sample_data(self, n_students):
        """Generate sample student data"""
        np.random.seed(42)
        
        data = {
            'student_id': [f'SAMPLE{1000 + i}' for i in range(n_students)],
            'name': [f'Student {i+1}' for i in range(n_students)],
            'previous_gpa': np.random.uniform(2.0, 4.0, n_students).round(2),
            'attendance_rate': np.random.beta(8, 2, n_students).round(3),
            'assignment_completion': np.random.beta(7, 3, n_students).round(3),
            'study_hours_weekly': np.random.randint(5, 25, n_students),
            'library_visits': np.random.poisson(3, n_students),
            'online_activity': np.random.beta(3, 2, n_students).round(3),
            'gender': np.random.choice([0, 1], n_students),
            'age': np.random.randint(18, 25, n_students),
            'midterm_score': np.random.randint(50, 95, n_students),
            'quiz_average': np.random.randint(55, 90, n_students),
            'sleep_hours': np.random.uniform(5.0, 9.0, n_students).round(1),
            'stress_level': np.random.randint(3, 9, n_students),
            'major': np.random.choice([
                'Computer Science', 'Engineering', 'Business', 
                'Biology', 'Mathematics', 'Psychology'
            ], n_students)
        }
        
        return pd.DataFrame(data)
    
    def add_missing_features(self, df):
        """Add missing features with default values"""
        required_features = [
            'previous_gpa', 'attendance_rate', 'assignment_completion',
            'study_hours_weekly', 'library_visits', 'online_activity',
            'gender', 'age', 'midterm_score', 'quiz_average',
            'sleep_hours', 'stress_level'
        ]
        
        for feature in required_features:
            if feature not in df.columns:
                if feature == 'gender':
                    df[feature] = 0
                elif feature == 'age':
                    df[feature] = 20
                elif feature in ['sleep_hours', 'stress_level']:
                    df[feature] = 7.0 if feature == 'sleep_hours' else 5
                else:
                    df[feature] = 0
        
        return df
    
    def predict_batch(self, batch_df):
        """Make predictions for batch of students"""
        results = []
        
        for idx, row in batch_df.iterrows():
            # Prepare features
            input_data = row.to_dict()
            features = self.prepare_features(input_data)
            
            # Make prediction
            prediction, probability = self.predict_single(features)
            
            # Calculate risk score
            risk_score = self.calculate_risk_score(input_data)
            
            # Store results
            results.append({
                'student_id': input_data.get('student_id', f'STU{idx}'),
                'name': input_data.get('name', f'Student {idx+1}'),
                'prediction': prediction,
                'prediction_label': 'High Score' if prediction == 1 else 'At Risk',
                'probability_high': probability[1],
                'probability_low': probability[0],
                'confidence': max(probability),
                'risk_score': risk_score,
                'previous_gpa': input_data.get('previous_gpa', 0),
                'attendance_rate': input_data.get('attendance_rate', 0),
                'study_hours_weekly': input_data.get('study_hours_weekly', 0),
                'midterm_score': input_data.get('midterm_score', 0),
                **input_data
            })
        
        return pd.DataFrame(results)
    
    def display_batch_summary(self, predictions_df):
        """Display batch prediction summary"""
        st.markdown("### 📈 Batch Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            high_score_count = (predictions_df['prediction'] == 1).sum()
            st.metric(
                "High Score Predictions",
                f"{high_score_count}",
                f"{high_score_count/len(predictions_df)*100:.1f}%"
            )
        
        with col2:
            at_risk_count = (predictions_df['prediction'] == 0).sum()
            st.metric(
                "At-Risk Students",
                f"{at_risk_count}",
                f"{-at_risk_count/len(predictions_df)*100:.1f}%"
            )
        
        with col3:
            avg_risk = predictions_df['risk_score'].mean()
            st.metric(
                "Average Risk Score",
                f"{avg_risk:.1f}",
                delta=None
            )
        
        with col4:
            avg_confidence = predictions_df['confidence'].mean() * 100
            st.metric(
                "Average Confidence",
                f"{avg_confidence:.1f}%",
                delta=None
            )
        
        # Quick visualization
        fig = px.pie(
            predictions_df,
            names='prediction_label',
            title='Prediction Distribution',
            color='prediction_label',
            color_discrete_map={'High Score': '#10B981', 'At Risk': '#EF4444'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def display_batch_predictions(self, predictions_df):
        """Display batch predictions in an interactive table"""
        # Create tabs for different views
        view_tabs = st.tabs(["All Students", "High Performers", "At-Risk Students", "Export Options"])
        
        with view_tabs[0]:
            display_cols = [
                'student_id', 'name', 'prediction_label', 'probability_high', 
                'risk_score', 'confidence', 'previous_gpa', 'attendance_rate',
                'study_hours_weekly', 'midterm_score'
            ]
            
            st.dataframe(
                predictions_df[display_cols].sort_values('risk_score', ascending=False),
                use_container_width=True,
                column_config={
                    "probability_high": st.column_config.ProgressColumn(
                        "High Score Probability",
                        format="%.1f%%",
                        min_value=0,
                        max_value=1.0
                    ),
                    "risk_score": st.column_config.ProgressColumn(
                        "Risk Score",
                        format="%.0f",
                        min_value=0,
                        max_value=100
                    ),
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.1f%%",
                        min_value=0,
                        max_value=1.0
                    )
                }
            )
        
        with view_tabs[1]:
            high_performers = predictions_df[predictions_df['prediction'] == 1]
            if not high_performers.empty:
                st.dataframe(
                    high_performers[display_cols].sort_values('probability_high', ascending=False),
                    use_container_width=True
                )
            else:
                st.info("No high performers found in this batch.")
        
        with view_tabs[2]:
            at_risk = predictions_df[predictions_df['prediction'] == 0]
            if not at_risk.empty:
                st.dataframe(
                    at_risk[display_cols].sort_values('risk_score', ascending=False),
                    use_container_width=True
                )
            else:
                st.success("🎉 No at-risk students identified in this batch!")
        
        with view_tabs[3]:
            st.markdown("### 💾 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export as CSV
                csv = predictions_df.to_csv(index=False)
                b64_csv = base64.b64encode(csv.encode()).decode()
                href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="batch_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">📥 Download CSV</a>'
                st.markdown(href_csv, unsafe_allow_html=True)
            
            with col2:
                # Export as Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    predictions_df.to_excel(writer, index=False, sheet_name='Predictions')
                b64_excel = base64.b64encode(output.getvalue()).decode()
                href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="batch_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">📊 Download Excel</a>'
                st.markdown(href_excel, unsafe_allow_html=True)
            
            with col3:
                # Generate summary report
                if st.button("📋 Generate Summary Report", use_container_width=True):
                    report = self.generate_batch_report(predictions_df)
                    b64_report = base64.b64encode(report.encode()).decode()
                    href_report = f'<a href="data:file/txt;base64,{b64_report}" download="batch_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt">📄 Download Report</a>'
                    st.markdown(href_report, unsafe_allow_html=True)
    
    def generate_batch_report(self, predictions_df):
        """Generate batch analysis report"""
        report = f"""
        BATCH PREDICTION ANALYSIS REPORT
        ================================
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Total Students Analyzed: {len(predictions_df)}
        
        PREDICTION OVERVIEW:
        - High Score Predictions: {(predictions_df['prediction'] == 1).sum()} ({(predictions_df['prediction'] == 1).sum()/len(predictions_df)*100:.1f}%)
        - At-Risk Students: {(predictions_df['prediction'] == 0).sum()} ({(predictions_df['prediction'] == 0).sum()/len(predictions_df)*100:.1f}%)
        - Average Risk Score: {predictions_df['risk_score'].mean():.1f}
        - Average Confidence: {predictions_df['confidence'].mean()*100:.1f}%
        
        PERFORMANCE STATISTICS:
        - Average GPA: {predictions_df['previous_gpa'].mean():.2f}
        - Average Attendance: {predictions_df['attendance_rate'].mean()*100:.1f}%
        - Average Study Hours: {predictions_df['study_hours_weekly'].mean():.1f}/week
        - Average Midterm Score: {predictions_df['midterm_score'].mean():.1f}
        
        TOP 5 HIGHEST RISK STUDENTS:
        """
        
        top_risk = predictions_df.nlargest(5, 'risk_score')
        for idx, student in top_risk.iterrows():
            report += f"\n{student.get('student_id', 'N/A')} - {student.get('name', 'Unknown')}:"
            report += f" Risk Score = {student['risk_score']:.1f}"
            report += f", GPA = {student['previous_gpa']:.2f}"
            report += f", Attendance = {student['attendance_rate']*100:.0f}%"
        
        report += f"""
        
        RECOMMENDED INTERVENTIONS:
        """
        
        # Add recommendations based on batch analysis
        if (predictions_df['attendance_rate'] < 0.8).mean() > 0.3:
            report += "\n- Implement attendance improvement program (over 30% below 80%)"
        
        if (predictions_df['study_hours_weekly'] < 10).mean() > 0.25:
            report += "\n- Organize study skills workshops (over 25% study less than 10 hrs/week)"
        
        if (predictions_df['stress_level'] > 6).mean() > 0.4:
            report += "\n- Provide stress management resources (over 40% report high stress)"
        
        return report
    
    def create_comparative_analysis_interface(self):
        """Create interface for comparative analysis"""
        st.markdown('<div class="sub-header">📈 Comparative Analysis</div>', unsafe_allow_html=True)
        
        # Check if batch predictions are available
        if st.session_state.batch_predictions is None or st.session_state.batch_predictions.empty:
            st.warning("""
            ⚠️ **Comparative Analysis requires batch data.**
            
            Please run a batch prediction first to enable comparative analysis.
            
            **Steps to enable:**
            1. Go to the **📊 Batch Student Prediction** tab
            2. Upload or enter data for multiple students
            3. Run the batch prediction
            4. Return to this tab for comparative analysis
            
            [Go to Batch Prediction](#batch-prediction)
            """)
            
            # Navigation button
            if st.button("📊 Go to Batch Prediction", use_container_width=True):
                # This would navigate to the batch prediction tab
                # In Streamlit, we can use st.query_params or session state
                st.session_state.current_tab = "Batch Prediction"
                st.rerun()
            
            return
        
        # If batch data is available, show comparative analysis
        predictions_df = st.session_state.batch_predictions
        
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                <div class="info-box">
                <h4>🔍 Compare and analyze multiple students to identify patterns, trends, and insights.</h4>
                <p>Use the tools below to perform comprehensive comparative analysis on your batch data.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #FEF3C7 0%, #FFFBEB 100%); border-radius: 10px;">
                <h4>📊 Analysis Ready</h4>
                <p>• Students loaded: {len(predictions_df)}</p>
                <p>• High performers: {(predictions_df['prediction'] == 1).sum()}</p>
                <p>• At-risk students: {(predictions_df['prediction'] == 0).sum()}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Comparative analysis tabs
        analysis_tabs = st.tabs([
            "📊 Performance Distribution", 
            "🎯 Feature Comparison", 
            "📈 Trend Analysis",
            "👥 Student Clusters",
            "🔍 Advanced Tools"
        ])
        
        with analysis_tabs[0]:
            self.plot_performance_distribution(predictions_df)
        
        with analysis_tabs[1]:
            self.plot_feature_comparison(predictions_df)
        
        with analysis_tabs[2]:
            self.plot_trend_analysis(predictions_df)
        
        with analysis_tabs[3]:
            self.plot_student_clusters(predictions_df)
        
        with analysis_tabs[4]:
            self.display_advanced_tools(predictions_df)
    
    def plot_performance_distribution(self, predictions_df):
        """Plot performance distribution visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution pie chart
            fig = px.pie(
                predictions_df,
                names='prediction_label',
                title='Prediction Distribution',
                color='prediction_label',
                color_discrete_map={'High Score': '#10B981', 'At Risk': '#EF4444'},
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk score distribution
            fig = px.histogram(
                predictions_df,
                x='risk_score',
                color='prediction_label',
                title='Risk Score Distribution',
                nbins=20,
                color_discrete_map={'High Score': '#10B981', 'At Risk': '#EF4444'},
                marginal="rug"
            )
            fig.update_layout(barmode='overlay')
            fig.update_traces(opacity=0.75)
            st.plotly_chart(fig, use_container_width=True)
        
        # GPA vs Risk scatter plot
        fig = px.scatter(
            predictions_df,
            x='previous_gpa',
            y='risk_score',
            color='prediction_label',
            size='confidence',
            hover_data=['student_id', 'name', 'midterm_score'],
            title='GPA vs Risk Score Analysis',
            color_discrete_map={'High Score': '#10B981', 'At Risk': '#EF4444'},
            trendline="ols"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_feature_comparison(self, predictions_df):
        """Plot feature comparison between groups"""
        st.markdown("### 🔍 Compare Features Between High Performers and At-Risk Students")
        
        # Select features to compare
        features_to_compare = st.multiselect(
            "Select features to compare:",
            ['previous_gpa', 'attendance_rate', 'assignment_completion', 
             'study_hours_weekly', 'midterm_score', 'quiz_average',
             'sleep_hours', 'stress_level', 'library_visits', 'online_activity'],
            default=['previous_gpa', 'attendance_rate', 'study_hours_weekly'],
            key='compare_features'
        )
        
        if features_to_compare:
            # Calculate statistics for each group
            high_performers = predictions_df[predictions_df['prediction'] == 1]
            at_risk = predictions_df[predictions_df['prediction'] == 0]
            
            comparison_data = []
            for feature in features_to_compare:
                if feature in predictions_df.columns:
                    comparison_data.append({
                        'Feature': feature.replace('_', ' ').title(),
                        'High Performers': high_performers[feature].mean(),
                        'At-Risk Students': at_risk[feature].mean(),
                        'Difference': high_performers[feature].mean() - at_risk[feature].mean(),
                        'P-Value': self.calculate_p_value(
                            high_performers[feature], 
                            at_risk[feature]
                        )
                    })
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                
                # Bar chart comparison
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='High Performers',
                    x=comp_df['Feature'],
                    y=comp_df['High Performers'],
                    marker_color='#10B981',
                    text=comp_df['High Performers'].round(2),
                    textposition='auto'
                ))
                
                fig.add_trace(go.Bar(
                    name='At-Risk Students',
                    x=comp_df['Feature'],
                    y=comp_df['At-Risk Students'],
                    marker_color='#EF4444',
                    text=comp_df['At-Risk Students'].round(2),
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title='Feature Comparison: High Performers vs At-Risk Students',
                    barmode='group',
                    height=500,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical significance table
                st.markdown("#### 📊 Statistical Significance")
                
                comp_df['Significance'] = comp_df['P-Value'].apply(
                    lambda x: '⭐ Highly Significant' if x < 0.01 else 
                             '✅ Significant' if x < 0.05 else 
                             '⚠️ Not Significant'
                )
                
                display_df = comp_df[['Feature', 'High Performers', 'At-Risk Students', 
                                     'Difference', 'P-Value', 'Significance']].copy()
                display_df = display_df.round(3)
                
                st.dataframe(
                    display_df.style.apply(
                        lambda x: ['background-color: #D1FAE5' if x['P-Value'] < 0.05 else '' for _ in x],
                        axis=1
                    ),
                    use_container_width=True
                )
    
    def calculate_p_value(self, group1, group2):
        """Calculate approximate p-value for two groups"""
        from scipy import stats
        try:
            _, p_value = stats.ttest_ind(group1.dropna(), group2.dropna())
            return p_value
        except:
            return 1.0
    
    def plot_trend_analysis(self, predictions_df):
        """Plot trend analysis"""
        st.markdown("### 📈 Trend Analysis")
        
        trend_tabs = st.tabs(["GPA Trends", "Study Habits", "Performance Patterns"])
        
        with trend_tabs[0]:
            if 'previous_gpa' in predictions_df.columns and 'midterm_score' in predictions_df.columns:
                fig = px.scatter(
                    predictions_df,
                    x='previous_gpa',
                    y='midterm_score',
                    color='prediction_label',
                    trendline='ols',
                    title='GPA vs Midterm Performance',
                    color_discrete_map={'High Score': '#10B981', 'At Risk': '#EF4444'},
                    hover_data=['student_id', 'name', 'attendance_rate']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with trend_tabs[1]:
            if 'study_hours_weekly' in predictions_df.columns and 'quiz_average' in predictions_df.columns:
                fig = px.scatter(
                    predictions_df,
                    x='study_hours_weekly',
                    y='quiz_average',
                    color='prediction_label',
                    size='attendance_rate',
                    title='Study Hours vs Quiz Performance',
                    color_discrete_map={'High Score': '#10B981', 'At Risk': '#EF4444'},
                    hover_data=['student_id', 'name', 'previous_gpa']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with trend_tabs[2]:
            # Performance progression
            if 'risk_score' in predictions_df.columns:
                sorted_df = predictions_df.sort_values('risk_score', ascending=False).reset_index()
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=sorted_df.index,
                    y=sorted_df['risk_score'],
                    mode='lines+markers',
                    name='Risk Score',
                    line=dict(color='#EF4444', width=2),
                    marker=dict(
                        size=8,
                        color=sorted_df['prediction'].map({1: '#10B981', 0: '#EF4444'}),
                        line=dict(color='white', width=1)
                    )
                ))
                
                # Add average lines
                fig.add_hline(
                    y=sorted_df['risk_score'].mean(),
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Average: {sorted_df['risk_score'].mean():.1f}"
                )
                
                fig.add_hline(
                    y=50,
                    line_dash="dot",
                    line_color="orange",
                    annotation_text="Risk Threshold: 50"
                )
                
                fig.update_layout(
                    title='Risk Score Progression (Sorted)',
                    xaxis_title='Student Rank',
                    yaxis_title='Risk Score',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def plot_student_clusters(self, predictions_df):
        """Plot student clustering"""
        st.markdown("### 👥 Student Clustering Analysis")
        
        # Select features for clustering
        cluster_features = st.multiselect(
            "Select features for clustering analysis:",
            ['previous_gpa', 'attendance_rate', 'study_hours_weekly', 
             'midterm_score', 'risk_score', 'stress_level',
             'assignment_completion', 'quiz_average', 'sleep_hours'],
            default=['previous_gpa', 'attendance_rate', 'study_hours_weekly'],
            key='cluster_features'
        )
        
        if len(cluster_features) >= 2:
            # Perform simple clustering (k-means like grouping)
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            
            # Prepare data
            cluster_data = predictions_df[cluster_features].fillna(0)
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            
            predictions_df['cluster'] = clusters
            
            # Visualize clusters
            if len(cluster_features) == 2:
                fig = px.scatter(
                    predictions_df,
                    x=cluster_features[0],
                    y=cluster_features[1],
                    color='cluster',
                    size='confidence',
                    hover_data=['student_id', 'name', 'prediction_label'],
                    title=f'Student Clusters: {cluster_features[0]} vs {cluster_features[1]}',
                    color_continuous_scale='Viridis'
                )
            else:
                # Use first three features for 3D plot
                fig = px.scatter_3d(
                    predictions_df,
                    x=cluster_features[0],
                    y=cluster_features[1],
                    z=cluster_features[2] if len(cluster_features) > 2 else cluster_features[0],
                    color='cluster',
                    size='confidence',
                    hover_data=['student_id', 'name', 'prediction_label'],
                    title='3D Student Clusters'
                )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster analysis
            st.markdown("#### 📊 Cluster Characteristics")
            
            cluster_stats = predictions_df.groupby('cluster').agg({
                'previous_gpa': 'mean',
                'attendance_rate': 'mean',
                'study_hours_weekly': 'mean',
                'risk_score': 'mean',
                'prediction': lambda x: (x == 1).mean()  # Percentage of high performers
            }).round(3)
            
            cluster_stats.columns = ['Avg GPA', 'Avg Attendance', 'Avg Study Hours', 
                                    'Avg Risk Score', 'High Performer %']
            
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Interpretation
            st.markdown("#### 💡 Cluster Interpretation")
            st.info("""
            **Cluster Analysis Guide:**
            - **Cluster 0**: Typically average performers with balanced metrics
            - **Cluster 1**: High-risk students needing intervention
            - **Cluster 2**: High performers with excellent metrics
            
            Use this analysis to identify groups of students with similar characteristics
            and tailor interventions accordingly.
            """)
        else:
            st.info("Please select at least 2 features for clustering analysis.")
    
    def display_advanced_tools(self, predictions_df):
        """Display advanced comparative analysis tools"""
        st.markdown("### 🔧 Advanced Comparative Tools")
        
        tool_tabs = st.tabs(["Student Comparison", "Correlation Matrix", "Predictive Insights"])
        
        with tool_tabs[0]:
            st.markdown("#### 🔍 Compare Specific Students")
            
            # Select students to compare
            student_options = []
            for idx, row in predictions_df.iterrows():
                label = f"{row['student_id']} - {row['name']} ({row['prediction_label']})"
                student_options.append((label, idx))
            
            selected_indices = st.multiselect(
                "Select students to compare (2-5 recommended):",
                options=[opt[1] for opt in student_options],
                format_func=lambda x: f"{predictions_df.loc[x, 'student_id']} - {predictions_df.loc[x, 'name']}",
                max_selections=5
            )
            
            if len(selected_indices) >= 2:
                selected_students = predictions_df.loc[selected_indices]
                
                # Comparison table
                comparison_features = [
                    'student_id', 'name', 'prediction_label', 'probability_high',
                    'risk_score', 'previous_gpa', 'attendance_rate', 
                    'study_hours_weekly', 'midterm_score', 'stress_level'
                ]
                
                st.dataframe(
                    selected_students[comparison_features].set_index('student_id').T,
                    use_container_width=True
                )
                
                # Radar chart comparison
                self.plot_comparison_radar(selected_students)
        
        with tool_tabs[1]:
            st.markdown("#### 📊 Feature Correlation Matrix")
            
            numeric_cols = predictions_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['cluster', 'prediction']]
            
            corr_matrix = predictions_df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                title='Feature Correlation Matrix',
                labels=dict(color="Correlation")
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Strongest correlations
            st.markdown("#### 🎯 Strongest Correlations")
            
            # Flatten correlation matrix
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': abs(corr_matrix.iloc[i, j]),
                        'Direction': 'Positive' if corr_matrix.iloc[i, j] > 0 else 'Negative'
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            top_correlations = corr_df.nlargest(10, 'Correlation')
            
            st.dataframe(top_correlations, use_container_width=True)
        
        with tool_tabs[2]:
            st.markdown("#### 🧠 Predictive Insights")
            
            # Identify key predictors
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': self.model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                st.markdown("##### 🎯 Top Predictive Features")
                st.dataframe(importance_df, use_container_width=True)
                
                # Insights based on data
                st.markdown("##### 💡 Data-Driven Insights")
                
                insights = []
                
                # GPA insight
                gpa_corr = predictions_df['previous_gpa'].corr(predictions_df['probability_high'])
                if abs(gpa_corr) > 0.3:
                    insights.append(f"• Previous GPA shows {'strong positive' if gpa_corr > 0 else 'strong negative'} correlation with predicted success (r={gpa_corr:.2f})")
                
                # Attendance insight
                attend_corr = predictions_df['attendance_rate'].corr(predictions_df['probability_high'])
                if abs(attend_corr) > 0.3:
                    insights.append(f"• Attendance rate is {'strongly correlated' if attend_corr > 0 else 'inversely correlated'} with performance (r={attend_corr:.2f})")
                
                # Study hours insight
                study_corr = predictions_df['study_hours_weekly'].corr(predictions_df['probability_high'])
                if abs(study_corr) > 0.2:
                    insights.append(f"• Study hours show {'moderate positive' if study_corr > 0 else 'moderate negative'} impact (r={study_corr:.2f})")
                
                # Stress insight
                stress_corr = predictions_df['stress_level'].corr(predictions_df['probability_high'])
                if stress_corr < -0.2:
                    insights.append(f"• Higher stress levels correlate with lower predicted performance (r={stress_corr:.2f})")
                
                if insights:
                    for insight in insights:
                        st.info(insight)
                else:
                    st.info("No strong correlations detected in this dataset.")
            
            # Recommendation engine
            st.markdown("##### 🎯 Personalized Recommendations Engine")
            
            if st.button("Generate Group Recommendations", use_container_width=True):
                recommendations = self.generate_group_recommendations(predictions_df)
                
                st.markdown("###### 📋 Recommended Interventions:")
                for i, rec in enumerate(recommendations, 1):
                    st.success(f"{i}. {rec}")
    
    def plot_comparison_radar(self, students_df):
        """Plot radar chart for student comparison"""
        features = ['previous_gpa', 'attendance_rate', 'assignment_completion',
                   'study_hours_weekly', 'midterm_score']
        
        # Normalize features
        normalized_data = []
        student_labels = []
        
        for idx, student in students_df.iterrows():
            student_data = []
            for feature in features:
                if feature in student:
                    # Normalize based on typical ranges
                    if feature == 'previous_gpa':
                        normalized = student[feature] / 4.0
                    elif feature in ['attendance_rate', 'assignment_completion']:
                        normalized = student[feature]
                    elif feature == 'study_hours_weekly':
                        normalized = student[feature] / 30
                    elif feature == 'midterm_score':
                        normalized = student[feature] / 100
                    else:
                        normalized = student[feature]
                    student_data.append(min(max(normalized, 0), 1))
            
            normalized_data.append(student_data)
            student_labels.append(f"{student['name']} ({student['prediction_label'][0]})")
        
        # Create radar chart
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, (label, data) in enumerate(zip(student_labels, normalized_data)):
            fig.add_trace(go.Scatterpolar(
                r=data + [data[0]],  # Close the polygon
                theta=features + [features[0]],
                fill='toself',
                name=label,
                line_color=colors[i % len(colors)],
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Student Comparison - Radar Chart",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def generate_group_recommendations(self, predictions_df):
        """Generate recommendations for the entire group"""
        recommendations = []
        
        # Analyze group statistics
        avg_gpa = predictions_df['previous_gpa'].mean()
        avg_attendance = predictions_df['attendance_rate'].mean()
        avg_study_hours = predictions_df['study_hours_weekly'].mean()
        avg_stress = predictions_df['stress_level'].mean()
        
        # Generate recommendations based on statistics
        if avg_gpa < 3.0:
            recommendations.append("Implement GPA improvement workshops focusing on study strategies")
        
        if avg_attendance < 0.8:
            recommendations.append(f"Launch attendance initiative (current average: {avg_attendance*100:.1f}%)")
        
        if avg_study_hours < 12:
            recommendations.append(f"Organize study skills training (average study hours: {avg_study_hours:.1f}/week)")
        
        if avg_stress > 6:
            recommendations.append(f"Provide stress management resources (average stress level: {avg_stress:.1f}/10)")
        
        # High-risk student interventions
        at_risk_count = (predictions_df['prediction'] == 0).sum()
        if at_risk_count > 0:
            recommendations.append(f"Prioritize interventions for {at_risk_count} at-risk students identified")
        
        # High performer opportunities
        high_performer_count = (predictions_df['prediction'] == 1).sum()
        if high_performer_count > 5:
            recommendations.append(f"Create peer tutoring program using {high_performer_count} high performers")
        
        return recommendations
    
    def display_system_features(self):
        """Display system feature links and information"""
        st.markdown("---")
        st.markdown("### 🛠️ System Features & Capabilities")
        
        # Create three columns for feature categories
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card info-card">
            <h4>🔬 System Information</h4>
            <p><strong>PSO-Optimized XGBoost</strong></p>
            <p>• Advanced machine learning optimized with Particle Swarm Intelligence</p>
            <p>• High accuracy predictions (90%+)</p>
            <p>• Real-time model updates</p>
            <p><a href="#model-info" style="color: #3B82F6; text-decoration: none; font-weight: 600;">📚 Learn More</a></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card info-card">
            <h4>📊 15+ Predictive Features</h4>
            <p>• Academic history and performance metrics</p>
            <p>• Study habits and behavioral patterns</p>
            <p>• Lifestyle factors and well-being indicators</p>
            <p><a href="#features" style="color: #3B82F6; text-decoration: none; font-weight: 600;">🔍 View Features</a></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card info-card">
            <h4>⚡ Real-time Analytics</h4>
            <p>• Instant predictions and recommendations</p>
            <p>• Live data visualization and dashboards</p>
            <p>• Continuous model performance monitoring</p>
            <p><a href="#analytics" style="color: #3B82F6; text-decoration: none; font-weight: 600;">📈 View Analytics</a></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card success-card">
            <h4>📁 Batch Capabilities</h4>
            <p><strong>CSV/Excel Upload</strong></p>
            <p>• Support for multiple file formats</p>
            <p>• Automatic data validation and cleaning</p>
            <p>• Template download available</p>
            <p><a href="#upload" style="color: #10B981; text-decoration: none; font-weight: 600;">📤 Upload Files</a></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card success-card">
            <h4>✍️ Manual Data Entry</h4>
            <p>• Interactive data tables</p>
            <p>• Real-time validation</p>
            <p>• Easy editing and updates</p>
            <p><a href="#manual-entry" style="color: #10B981; text-decoration: none; font-weight: 600;">📝 Enter Data</a></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card success-card">
            <h4>🎯 Sample Data Generation</h4>
            <p>• Generate realistic test data</p>
            <p>• Customizable sample sizes</p>
            <p>• Realistic patterns and correlations</p>
            <p><a href="#sample-data" style="color: #10B981; text-decoration: none; font-weight: 600;">🎲 Generate Data</a></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="card warning-card">
            <h4>📈 Comparative Analysis</h4>
            <p>• Side-by-side student comparison</p>
            <p>• Group performance analysis</p>
            <p>• Performance gap identification</p>
            <p><a href="#comparative-analysis" style="color: #F59E0B; text-decoration: none; font-weight: 600;">🔍 Compare Now</a></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card warning-card">
            <h4>📊 Trend Identification</h4>
            <p>• Performance trend analysis</p>
            <p>• Predictive trend forecasting</p>
            <p>• Historical pattern recognition</p>
            <p><a href="#trends" style="color: #F59E0B; text-decoration: none; font-weight: 600;">📈 View Trends</a></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card warning-card">
            <h4>👥 Cluster Detection</h4>
            <p>• Automated student grouping</p>
            <p>• Similarity-based clustering</p>
            <p>• Targeted intervention planning</p>
            <p><a href="#clusters" style="color: #F59E0B; text-decoration: none; font-weight: 600;">🔍 Detect Clusters</a></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick access buttons
        st.markdown("---")
        st.markdown("### 🚀 Quick Access")
        
        quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
        
        with quick_col1:
            if st.button("📚 Model Info", use_container_width=True):
                st.session_state.show_model_info = True
        
        with quick_col2:
            if st.button("📤 Upload Data", use_container_width=True):
                st.session_state.current_tab = "Batch Prediction"
        
        with quick_col3:
            if st.button("🔍 Compare", use_container_width=True):
                st.session_state.current_tab = "Comparative Analysis"
        
        with quick_col4:
            if st.button("📈 Analytics", use_container_width=True):
                st.session_state.show_analytics = True
    
    def run(self):
        """Main application runner"""
        # Main header
        st.markdown('<h1 class="main-header">🎓 Advanced Student Analytics System</h1>', unsafe_allow_html=True)
        st.markdown("### Powered by PSO-Optimized Machine Learning & Advanced Analytics")
        
        # Navigation tabs
        tab1, tab2, tab3 = st.tabs([
            "🎯 Single Student Prediction", 
            "📊 Batch Student Prediction", 
            "📈 Comparative Analysis"
        ])
        
        # Run each interface
        with tab1:
            predict_button, input_data = self.create_single_student_interface()
            
            if predict_button:
                with st.spinner("🔮 Analyzing student data..."):
                    prediction, probability, features = self.process_single_prediction(input_data)
                    self.display_single_prediction_results(prediction, probability, features, input_data)
        
        with tab2:
            self.create_batch_prediction_interface()
        
        with tab3:
            self.create_comparative_analysis_interface()
        
        # Display system features
        self.display_system_features()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6B7280; padding: 20px;">
        <p>Advanced Student Analytics System v2.0 | Powered by PSO-Optimized Machine Learning</p>
        <p>📧 Contact: analytics@university.edu | 📞 Support: +1 (555) 123-4567</p>
        <p>© 2024 Educational Analytics Platform. All rights reserved.</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function"""
    # Initialize and run the system
    system = AdvancedStudentAnalyticsSystem()
    system.run()

if __name__ == "__main__":
    main()