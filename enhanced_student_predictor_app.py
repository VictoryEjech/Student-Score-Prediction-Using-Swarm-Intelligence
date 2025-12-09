# enhanced_student_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
from typing import List, Dict, Tuple
import base64
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
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
    }
    .success-card {
        border-left: 5px solid #10B981;
    }
    .warning-card {
        border-left: 5px solid #F59E0B;
    }
    .danger-card {
        border-left: 5px solid #EF4444;
    }
    .info-card {
        border-left: 5px solid #3B82F6;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3B82F6, #1D4ED8);
        color: white;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
    }
    .tab-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .highlight-row {
        background-color: #FEF3C7 !important;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedStudentPredictorApp:
    def __init__(self):
        """Initialize the app and load models"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.batch_results = None
        self.comparison_data = None
        self.load_models()
        
    def load_models(self):
        """Load trained model and scaler"""
        try:
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
            
            st.sidebar.success("✅ Models loaded successfully!")
            
        except FileNotFoundError:
            st.sidebar.warning("⚠️ Model files not found. Using simulated predictions.")
            self.model = None
            self.scaler = None
    
    def prepare_features(self, input_data: Dict) -> pd.DataFrame:
        """Prepare and validate input features for single student"""
        features = pd.DataFrame([input_data])
        
        # Create interaction features
        features['gpa_attendance_interaction'] = (
            features['previous_gpa'] * features['attendance_rate']
        )
        features['study_completion_interaction'] = (
            features['study_hours_weekly'] * features['assignment_completion']
        )
        features['sleep_study_interaction'] = (
            features['sleep_hours'] * features['study_hours_weekly'] / 10
        )
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in features.columns:
                features[feature] = 0
        
        features = features[self.feature_names]
        return features
    
    def prepare_batch_features(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for batch prediction"""
        features_df = batch_df.copy()
        
        # Create interaction features
        features_df['gpa_attendance_interaction'] = (
            features_df['previous_gpa'] * features_df['attendance_rate']
        )
        features_df['study_completion_interaction'] = (
            features_df['study_hours_weekly'] * features_df['assignment_completion']
        )
        features_df['sleep_study_interaction'] = (
            features_df['sleep_hours'] * features_df['study_hours_weekly'] / 10
        )
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Reorder columns
        features_df = features_df[self.feature_names]
        
        return features_df
    
    def predict_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for a batch of students"""
        if batch_df.empty:
            return pd.DataFrame()
        
        # Prepare features
        features = self.prepare_batch_features(batch_df)
        
        if self.model is None or self.scaler is None:
            # Simulate predictions
            predictions = []
            probabilities = []
            
            for idx, row in features.iterrows():
                prob = min(0.95, max(0.05, 
                    0.3 * (row['previous_gpa'] - 2.0) / 2.0 +
                    0.2 * (row['attendance_rate'] - 0.5) / 0.5 +
                    0.15 * (row['study_hours_weekly'] - 10) / 20 +
                    0.1 * np.random.random()
                ))
                predictions.append(1 if prob > 0.5 else 0)
                probabilities.append([1-prob, prob])
        else:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make predictions
            predictions = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)
        
        # Create results dataframe
        results_df = batch_df.copy()
        results_df['prediction'] = predictions
        results_df['prediction_label'] = ['High Score' if p == 1 else 'At Risk' for p in predictions]
        results_df['probability_high'] = [prob[1] for prob in probabilities]
        results_df['probability_low'] = [prob[0] for prob in probabilities]
        results_df['confidence'] = [max(prob) for prob in probabilities]
        results_df['confidence_level'] = [
            'High' if c > 0.8 else 'Medium' if c > 0.6 else 'Low' 
            for c in results_df['confidence']
        ]
        
        # Calculate risk score (0-100)
        results_df['risk_score'] = results_df.apply(
            lambda row: self.calculate_risk_score(row), axis=1
        )
        
        # Add recommendations
        results_df['recommendations'] = results_df.apply(
            lambda row: self.get_batch_recommendations(row), axis=1
        )
        
        return results_df
    
    def calculate_risk_score(self, student_data: pd.Series) -> float:
        """Calculate comprehensive risk score (0-100)"""
        risk_factors = []
        
        # Academic risks
        if student_data['previous_gpa'] < 2.5:
            risk_factors.append(30)
        elif student_data['previous_gpa'] < 3.0:
            risk_factors.append(15)
        
        if student_data['attendance_rate'] < 0.7:
            risk_factors.append(25)
        elif student_data['attendance_rate'] < 0.8:
            risk_factors.append(12)
        
        if student_data['assignment_completion'] < 0.8:
            risk_factors.append(20)
        
        # Behavioral risks
        if student_data['study_hours_weekly'] < 10:
            risk_factors.append(15)
        
        if student_data['stress_level'] > 7:
            risk_factors.append(20)
        
        if student_data['sleep_hours'] < 6:
            risk_factors.append(15)
        
        # Calculate weighted risk
        risk_score = min(100, sum(risk_factors))
        return risk_score
    
    def get_batch_recommendations(self, student_data: pd.Series) -> str:
        """Generate concise recommendations for batch processing"""
        recommendations = []
        
        if student_data['attendance_rate'] < 0.8:
            recommendations.append("Improve attendance")
        
        if student_data['assignment_completion'] < 0.9:
            recommendations.append("Complete assignments")
        
        if student_data['study_hours_weekly'] < 12:
            recommendations.append("Increase study time")
        
        if student_data['stress_level'] > 6:
            recommendations.append("Manage stress")
        
        if student_data['sleep_hours'] < 7:
            recommendations.append("Get more sleep")
        
        return "; ".join(recommendations) if recommendations else "Maintain current habits"
    
    def create_batch_input_interface(self) -> Tuple[bool, pd.DataFrame]:
        """Create interface for batch input"""
        st.markdown('<div class="card info-card">', unsafe_allow_html=True)
        st.markdown("### 📥 Batch Data Input")
        
        input_method = st.radio(
            "Choose input method:",
            ["📁 Upload CSV/Excel File", "📝 Enter Data Manually", "🎯 Use Sample Data"],
            horizontal=True
        )
        
        batch_df = pd.DataFrame()
        
        if input_method == "📁 Upload CSV/Excel File":
            uploaded_file = st.file_uploader(
                "Upload student data file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload a CSV or Excel file with student data"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        batch_df = pd.read_csv(uploaded_file)
                    else:
                        batch_df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ Successfully loaded {len(batch_df)} student records")
                    
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        elif input_method == "📝 Enter Data Manually":
            num_students = st.number_input(
                "Number of students to enter:",
                min_value=1,
                max_value=100,
                value=5,
                step=1
            )
            
            if num_students > 0:
                # Create empty dataframe with required columns
                template_df = pd.DataFrame({
                    'student_id': [f'STU{1000 + i}' for i in range(num_students)],
                    'name': [f'Student {i+1}' for i in range(num_students)],
                    'previous_gpa': 3.0,
                    'attendance_rate': 0.8,
                    'assignment_completion': 0.85,
                    'study_hours_weekly': 15,
                    'midterm_score': 75,
                    'quiz_average': 72
                })
                
                # Use data editor for manual entry
                st.info("Edit the table below with student data:")
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
                        )
                    }
                )
                
                batch_df = edited_df.copy()
        
        else:  # Use Sample Data
            sample_size = st.slider("Sample size:", 5, 50, 20)
            batch_df = self.generate_sample_data(sample_size)
            st.info(f"✅ Generated {sample_size} sample student records")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show preview if data exists
        if not batch_df.empty:
            st.markdown("#### 👁️ Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            # Add missing features with default values
            batch_df = self.add_missing_features(batch_df)
            
            return True, batch_df
        else:
            return False, pd.DataFrame()
    
    def generate_sample_data(self, n_students: int) -> pd.DataFrame:
        """Generate sample student data for testing"""
        np.random.seed(42)
        
        data = {
            'student_id': [f'SAMPLE{1000 + i}' for i in range(n_students)],
            'name': [f'Student {i+1}' for i in range(n_students)],
            'previous_gpa': np.random.uniform(2.0, 4.0, n_students).round(2),
            'attendance_rate': np.random.beta(8, 2, n_students).round(2),
            'assignment_completion': np.random.beta(7, 3, n_students).round(2),
            'study_hours_weekly': np.random.randint(5, 25, n_students),
            'library_visits': np.random.poisson(3, n_students),
            'online_activity': np.random.beta(3, 2, n_students).round(2),
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
    
    def add_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
    
    def display_batch_results(self, results_df: pd.DataFrame):
        """Display batch prediction results"""
        st.markdown('<div class="card success-card">', unsafe_allow_html=True)
        st.markdown(f"### 📊 Batch Prediction Results: {len(results_df)} Students")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            high_score_count = (results_df['prediction'] == 1).sum()
            st.metric(
                "High Score Predictions",
                f"{high_score_count}",
                f"{high_score_count/len(results_df)*100:.1f}%"
            )
        
        with col2:
            at_risk_count = (results_df['prediction'] == 0).sum()
            st.metric(
                "At-Risk Students",
                f"{at_risk_count}",
                f"{-at_risk_count/len(results_df)*100:.1f}%"
            )
        
        with col3:
            avg_risk = results_df['risk_score'].mean()
            st.metric(
                "Average Risk Score",
                f"{avg_risk:.1f}",
                delta=None
            )
        
        with col4:
            avg_confidence = results_df['confidence'].mean() * 100
            st.metric(
                "Average Confidence",
                f"{avg_confidence:.1f}%",
                delta=None
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed results table
        st.markdown("#### 📋 Detailed Predictions")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["All Results", "At-Risk Students", "High Performers"])
        
        with tab1:
            display_cols = [
                'student_id', 'name', 'prediction_label', 'probability_high', 
                'risk_score', 'confidence_level', 'recommendations'
            ]
            if 'major' in results_df.columns:
                display_cols.insert(2, 'major')
            
            st.dataframe(
                results_df[display_cols].sort_values('risk_score', ascending=False),
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
                    )
                }
            )
        
        with tab2:
            at_risk_df = results_df[results_df['prediction'] == 0]
            if not at_risk_df.empty:
                st.dataframe(
                    at_risk_df[display_cols].sort_values('risk_score', ascending=False),
                    use_container_width=True
                )
            else:
                st.success("🎉 No at-risk students identified!")
        
        with tab3:
            high_performers_df = results_df[results_df['prediction'] == 1]
            if not high_performers_df.empty:
                st.dataframe(
                    high_performers_df[display_cols].sort_values('probability_high', ascending=False),
                    use_container_width=True
                )
        
        # Download options
        st.markdown("#### 💾 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download as CSV", use_container_width=True):
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Click to download",
                    data=csv,
                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Download Summary Report", use_container_width=True):
                summary = self.generate_summary_report(results_df)
                st.download_button(
                    label="Click to download",
                    data=summary,
                    file_name=f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("🔄 Run New Batch", use_container_width=True):
                st.experimental_rerun()
    
    def generate_summary_report(self, results_df: pd.DataFrame) -> str:
        """Generate a text summary report"""
        report = f"""
        BATCH PREDICTION SUMMARY REPORT
        ================================
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Total Students Analyzed: {len(results_df)}
        
        PREDICTION OVERVIEW:
        - High Score Predictions: {(results_df['prediction'] == 1).sum()} ({(results_df['prediction'] == 1).sum()/len(results_df)*100:.1f}%)
        - At-Risk Students: {(results_df['prediction'] == 0).sum()} ({(results_df['prediction'] == 0).sum()/len(results_df)*100:.1f}%)
        - Average Risk Score: {results_df['risk_score'].mean():.1f}
        - Average Confidence: {results_df['confidence'].mean()*100:.1f}%
        
        TOP 5 HIGHEST RISK STUDENTS:
        """
        
        top_risk = results_df.nlargest(5, 'risk_score')
        for idx, student in top_risk.iterrows():
            report += f"\n{student.get('student_id', 'N/A')}: Risk Score = {student['risk_score']:.1f}"
            report += f", Probability of High Score = {student['probability_high']*100:.1f}%"
            if 'name' in student:
                report += f", Name = {student['name']}"
        
        report += f"""
        
        KEY STATISTICS:
        - Average GPA: {results_df['previous_gpa'].mean():.2f}
        - Average Attendance: {results_df['attendance_rate'].mean()*100:.1f}%
        - Average Study Hours: {results_df['study_hours_weekly'].mean():.1f}/week
        - Average Midterm Score: {results_df['midterm_score'].mean():.1f}
        
        RECOMMENDATIONS:
        """
        
        # Common recommendations
        if (results_df['attendance_rate'] < 0.8).mean() > 0.3:
            report += "\n- Consider campus-wide attendance improvement initiatives"
        
        if (results_df['study_hours_weekly'] < 10).mean() > 0.25:
            report += "\n- Implement study skills workshops"
        
        if (results_df['stress_level'] > 6).mean() > 0.4:
            report += "\n- Provide stress management resources and counseling"
        
        return report
    
    def create_comparative_analysis(self, results_df: pd.DataFrame):
        """Create comprehensive comparative analysis"""
        st.markdown('<div class="sub-header">📈 Comparative Analysis Dashboard</div>', unsafe_allow_html=True)
        
        if results_df is None or results_df.empty:
            st.warning("No data available for comparative analysis.")
            return
        
        # Overview metrics
        st.markdown('<div class="card info-card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Students Compared", len(results_df))
        
        with col2:
            performance_gap = (
                results_df[results_df['prediction'] == 1]['previous_gpa'].mean() - 
                results_df[results_df['prediction'] == 0]['previous_gpa'].mean()
            )
            st.metric("GPA Performance Gap", f"{performance_gap:.2f}")
        
        with col3:
            avg_high_score_gpa = results_df[results_df['prediction'] == 1]['previous_gpa'].mean()
            st.metric("Avg GPA (High Performers)", f"{avg_high_score_gpa:.2f}")
        
        with col4:
            avg_risk_gpa = results_df[results_df['prediction'] == 0]['previous_gpa'].mean()
            st.metric("Avg GPA (At-Risk)", f"{avg_risk_gpa:.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Main comparative analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Performance Distribution", 
            "🎯 Feature Comparison", 
            "📈 Trend Analysis",
            "👥 Student Clusters"
        ])
        
        with tab1:
            self.plot_performance_distribution(results_df)
        
        with tab2:
            self.plot_feature_comparison(results_df)
        
        with tab3:
            self.plot_trend_analysis(results_df)
        
        with tab4:
            self.plot_student_clusters(results_df)
    
    def plot_performance_distribution(self, results_df: pd.DataFrame):
        """Plot performance distribution visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution pie chart
            fig = px.pie(
                results_df,
                names='prediction_label',
                title='Prediction Distribution',
                color='prediction_label',
                color_discrete_map={'High Score': '#10B981', 'At Risk': '#EF4444'},
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk score distribution
            fig = px.histogram(
                results_df,
                x='risk_score',
                color='prediction_label',
                title='Risk Score Distribution',
                nbins=20,
                color_discrete_map={'High Score': '#10B981', 'At Risk': '#EF4444'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # GPA vs Risk scatter plot
        fig = px.scatter(
            results_df,
            x='previous_gpa',
            y='risk_score',
            color='prediction_label',
            size='confidence',
            hover_data=['student_id', 'name', 'midterm_score'],
            title='GPA vs Risk Score Analysis',
            color_discrete_map={'High Score': '#10B981', 'At Risk': '#EF4444'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_feature_comparison(self, results_df: pd.DataFrame):
        """Plot feature comparison between high performers and at-risk students"""
        
        # Select features to compare
        features_to_compare = st.multiselect(
            "Select features to compare:",
            ['previous_gpa', 'attendance_rate', 'assignment_completion', 
             'study_hours_weekly', 'midterm_score', 'quiz_average',
             'sleep_hours', 'stress_level'],
            default=['previous_gpa', 'attendance_rate', 'study_hours_weekly']
        )
        
        if features_to_compare:
            # Calculate averages for each group
            high_performers = results_df[results_df['prediction'] == 1]
            at_risk = results_df[results_df['prediction'] == 0]
            
            comparison_data = []
            for feature in features_to_compare:
                if feature in results_df.columns:
                    comparison_data.append({
                        'Feature': feature,
                        'High Performers': high_performers[feature].mean(),
                        'At-Risk Students': at_risk[feature].mean(),
                        'Difference': high_performers[feature].mean() - at_risk[feature].mean()
                    })
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                
                # Bar chart comparison
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='High Performers',
                    x=comp_df['Feature'],
                    y=comp_df['High Performers'],
                    marker_color='#10B981'
                ))
                
                fig.add_trace(go.Bar(
                    name='At-Risk Students',
                    x=comp_df['Feature'],
                    y=comp_df['At-Risk Students'],
                    marker_color='#EF4444'
                ))
                
                fig.update_layout(
                    title='Feature Comparison: High Performers vs At-Risk Students',
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display difference table
                st.markdown("#### 📋 Performance Gap Analysis")
                st.dataframe(
                    comp_df.style.format({
                        'High Performers': '{:.2f}',
                        'At-Risk Students': '{:.2f}',
                        'Difference': '{:.2f}'
                    }).background_gradient(
                        subset=['Difference'], 
                        cmap='RdYlGn_r'
                    ),
                    use_container_width=True
                )
    
    def plot_trend_analysis(self, results_df: pd.DataFrame):
        """Plot trend analysis across different metrics"""
        
        # Create tabs for different trend analyses
        trend_tab1, trend_tab2, trend_tab3 = st.tabs(["GPA Trends", "Study Habits", "Risk Patterns"])
        
        with trend_tab1:
            if 'previous_gpa' in results_df.columns and 'midterm_score' in results_df.columns:
                fig = px.scatter(
                    results_df,
                    x='previous_gpa',
                    y='midterm_score',
                    color='prediction_label',
                    trendline='ols',
                    title='GPA vs Midterm Performance',
                    color_discrete_map={'High Score': '#10B981', 'At Risk': '#EF4444'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with trend_tab2:
            if 'study_hours_weekly' in results_df.columns and 'quiz_average' in results_df.columns:
                fig = px.scatter(
                    results_df,
                    x='study_hours_weekly',
                    y='quiz_average',
                    color='prediction_label',
                    size='attendance_rate',
                    title='Study Hours vs Quiz Performance',
                    color_discrete_map={'High Score': '#10B981', 'At Risk': '#EF4444'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with trend_tab3:
            # Risk progression analysis
            if 'risk_score' in results_df.columns:
                # Sort by risk score
                sorted_df = results_df.sort_values('risk_score', ascending=False)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(sorted_df))),
                    y=sorted_df['risk_score'],
                    mode='lines+markers',
                    name='Risk Score',
                    line=dict(color='#EF4444', width=2)
                ))
                
                # Add average line
                fig.add_hline(
                    y=sorted_df['risk_score'].mean(),
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Average: {sorted_df['risk_score'].mean():.1f}"
                )
                
                fig.update_layout(
                    title='Risk Score Distribution (Sorted)',
                    xaxis_title='Student Rank',
                    yaxis_title='Risk Score',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def plot_student_clusters(self, results_df: pd.DataFrame):
        """Plot student clustering based on key features"""
        
        # Select features for clustering
        cluster_features = st.multiselect(
            "Select features for clustering:",
            ['previous_gpa', 'attendance_rate', 'study_hours_weekly', 
             'midterm_score', 'risk_score', 'stress_level'],
            default=['previous_gpa', 'attendance_rate', 'study_hours_weekly'],
            key='cluster_features'
        )
        
        if len(cluster_features) >= 2:
            # Create 2D/3D scatter plot based on selected features
            if len(cluster_features) == 2:
                fig = px.scatter(
                    results_df,
                    x=cluster_features[0],
                    y=cluster_features[1],
                    color='prediction_label',
                    size='confidence',
                    hover_data=['student_id', 'name'],
                    title=f'Student Clusters: {cluster_features[0]} vs {cluster_features[1]}',
                    color_discrete_map={'High Score': '#10B981', 'At Risk': '#EF4444'}
                )
            else:
                fig = px.scatter_3d(
                    results_df,
                    x=cluster_features[0],
                    y=cluster_features[1],
                    z=cluster_features[2],
                    color='prediction_label',
                    size='confidence',
                    hover_data=['student_id', 'name'],
                    title=f'3D Student Clusters',
                    color_discrete_map={'High Score': '#10B981', 'At Risk': '#EF4444'}
                )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster statistics
            st.markdown("#### 📊 Cluster Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Calculate correlation matrix for selected features
                corr_matrix = results_df[cluster_features].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu',
                    title='Feature Correlation Matrix'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Summary statistics
                st.dataframe(
                    results_df[cluster_features].describe().round(2),
                    use_container_width=True
                )
        else:
            st.info("Please select at least 2 features for clustering analysis.")
    
    def create_dashboard_interface(self):
        """Create the main dashboard interface"""
        # Header
        st.markdown('<h1 class="main-header">🎓 Advanced Student Analytics System</h1>', unsafe_allow_html=True)
        st.markdown("### Powered by PSO-Optimized Machine Learning & Batch Intelligence")
        
        # Navigation
        st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <span style='background-color: #3B82F6; color: white; padding: 10px 20px; border-radius: 5px; margin: 0 10px;'>
                🎯 Single Prediction
            </span>
            <span style='background-color: #10B981; color: white; padding: 10px 20px; border-radius: 5px; margin: 0 10px;'>
                📊 Batch Analysis
            </span>
            <span style='background-color: #8B5CF6; color: white; padding: 10px 20px; border-radius: 5px; margin: 0 10px;'>
                📈 Comparative Insights
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different modes
        tab1, tab2, tab3 = st.tabs([
            "🎯 Single Student Prediction", 
            "📊 Batch Prediction", 
            "📈 Comparative Analysis"
        ])
        
        return tab1, tab2, tab3
    
    def run_single_prediction(self, tab):
        """Run single student prediction interface"""
        with tab:
            # Keep existing single prediction interface
            st.markdown("### 🎯 Single Student Prediction")
            st.markdown("Enter individual student details for personalized prediction and recommendations.")
            
            # Your existing single prediction form and logic here
            # (This would be your original single prediction interface)
            
            st.info("Single prediction interface would go here...")
    
    def run_batch_prediction(self, tab):
        """Run batch prediction interface"""
        with tab:
            st.markdown("### 📊 Batch Student Prediction")
            st.markdown("Upload or enter data for multiple students to analyze performance at scale.")
            
            # Batch input interface
            has_data, batch_df = self.create_batch_input_interface()
            
            if has_data and not batch_df.empty:
                if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                    with st.spinner("🔮 Analyzing student batch..."):
                        self.batch_results = self.predict_batch(batch_df)
                        
                        # Display results
                        self.display_batch_results(self.batch_results)
                        
                        # Store for comparative analysis
                        self.comparison_data = self.batch_results
    
    def run_comparative_analysis(self, tab):
        """Run comparative analysis interface"""
        with tab:
            st.markdown("### 📈 Comparative Analysis")
            st.markdown("Compare and analyze multiple students to identify patterns, trends, and insights.")
            
            if self.comparison_data is not None and not self.comparison_data.empty:
                self.create_comparative_analysis(self.comparison_data)
                
                # Additional comparison tools
                st.markdown("---")
                st.markdown("### 🎯 Advanced Comparison Tools")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔍 Compare Selected Students", use_container_width=True):
                        self.show_student_comparison_tool()
                
                with col2:
                    if st.button("📋 Generate Group Report", use_container_width=True):
                        self.generate_group_report()
            else:
                st.info("👈 Run a batch prediction first to enable comparative analysis.")
                st.markdown("""
                **Features available after batch prediction:**
                - Performance distribution analysis
                - Feature comparison between high performers and at-risk students
                - Trend analysis across different metrics
                - Student clustering based on key characteristics
                - Advanced comparison tools for selected students
                """)
    
    def show_student_comparison_tool(self):
        """Show tool for comparing specific students"""
        if self.comparison_data is not None and not self.comparison_data.empty:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🔍 Compare Specific Students")
            
            # Let user select students to compare
            student_options = list(zip(
                self.comparison_data['student_id'].astype(str) + " - " + 
                self.comparison_data.get('name', 'Unknown').astype(str),
                self.comparison_data.index
            ))
            
            selected_indices = st.multiselect(
                "Select students to compare:",
                options=[opt[1] for opt in student_options],
                format_func=lambda x: f"{self.comparison_data.loc[x, 'student_id']} - {self.comparison_data.loc[x, 'name']}"
            )
            
            if len(selected_indices) >= 2:
                selected_students = self.comparison_data.loc[selected_indices]
                
                # Create comparison table
                comparison_features = [
                    'student_id', 'name', 'prediction_label', 'probability_high',
                    'risk_score', 'previous_gpa', 'attendance_rate', 
                    'study_hours_weekly', 'midterm_score'
                ]
                
                st.dataframe(
                    selected_students[comparison_features].T,
                    use_container_width=True
                )
                
                # Create radar chart for comparison
                self.plot_student_radar_comparison(selected_students)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def plot_student_radar_comparison(self, students_df: pd.DataFrame):
        """Plot radar chart comparing selected students"""
        features = ['previous_gpa', 'attendance_rate', 'assignment_completion',
                   'study_hours_weekly', 'midterm_score', 'risk_score']
        
        # Normalize features for radar chart
        normalized_data = []
        student_names = []
        
        for idx, student in students_df.iterrows():
            student_data = []
            for feature in features:
                if feature in student:
                    # Normalize to 0-1 scale
                    if feature == 'risk_score':
                        # Invert risk score for positive display
                        normalized = 1 - (student[feature] / 100)
                    else:
                        if feature == 'previous_gpa':
                            normalized = student[feature] / 4.0
                        elif feature in ['attendance_rate', 'assignment_completion']:
                            normalized = student[feature]
                        elif feature in ['study_hours_weekly']:
                            normalized = student[feature] / 30
                        elif feature in ['midterm_score']:
                            normalized = student[feature] / 100
                        else:
                            normalized = student[feature]
                    student_data.append(min(max(normalized, 0), 1))
            
            normalized_data.append(student_data)
            student_names.append(student.get('name', f"Student {student['student_id']}"))
        
        # Create radar chart
        fig = go.Figure()
        
        for i, (student_name, data) in enumerate(zip(student_names, normalized_data)):
            fig.add_trace(go.Scatterpolar(
                r=data + [data[0]],  # Close the polygon
                theta=features + [features[0]],
                fill='toself',
                name=student_name
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
    
    def generate_group_report(self):
        """Generate comprehensive group report"""
        if self.comparison_data is not None:
            report = f"""
            COMPREHENSIVE GROUP ANALYSIS REPORT
            ===================================
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Total Students: {len(self.comparison_data)}
            
            OVERALL PERFORMANCE:
            - High Performers: {(self.comparison_data['prediction'] == 1).sum()}
            - At-Risk Students: {(self.comparison_data['prediction'] == 0).sum()}
            - Average Risk Score: {self.comparison_data['risk_score'].mean():.1f}
            
            KEY INSIGHTS:
            """
            
            # Add insights based on data
            if (self.comparison_data['attendance_rate'] < 0.8).mean() > 0.3:
                report += "\n1. Attendance Issue: Over 30% of students have attendance below 80%"
            
            if (self.comparison_data['study_hours_weekly'] < 10).mean() > 0.25:
                report += "\n2. Study Time Deficit: 25%+ of students study less than 10 hours/week"
            
            if (self.comparison_data['stress_level'] > 6).mean() > 0.4:
                report += "\n3. High Stress Levels: 40%+ of students report stress levels above 6/10"
            
            report += "\n\nRECOMMENDED INTERVENTIONS:\n"
            
            # Generate recommendations
            recommendations = []
            
            avg_gap = (
                self.comparison_data[self.comparison_data['prediction'] == 1]['previous_gpa'].mean() -
                self.comparison_data[self.comparison_data['prediction'] == 0]['previous_gpa'].mean()
            )
            
            if avg_gap > 0.5:
                recommendations.append(f"- Implement GPA improvement program (current gap: {avg_gap:.2f})")
            
            if self.comparison_data['attendance_rate'].mean() < 0.85:
                recommendations.append(f"- Launch attendance initiative (current average: {self.comparison_data['attendance_rate'].mean()*100:.1f}%)")
            
            if self.comparison_data['study_hours_weekly'].mean() < 15:
                recommendations.append(f"- Organize study skills workshops (current average: {self.comparison_data['study_hours_weekly'].mean():.1f} hours/week)")
            
            report += "\n".join(recommendations) if recommendations else "- No major interventions needed at this time"
            
            # Display report
            st.markdown("### 📋 Group Analysis Report")
            st.text_area("Report Content", report, height=300)
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"group_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    def run(self):
        """Main application runner"""
        # Create dashboard interface
        tab1, tab2, tab3 = self.create_dashboard_interface()
        
        # Run different prediction modes
        self.run_single_prediction(tab1)
        self.run_batch_prediction(tab2)
        self.run_comparative_analysis(tab3)
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**System Information**")
            st.markdown("- PSO-Optimized XGBoost")
            st.markdown("- 15+ Predictive Features")
            st.markdown("- Real-time Analytics")
        
        with col2:
            st.markdown("**Batch Capabilities**")
            st.markdown("- CSV/Excel Upload")
            st.markdown("- Manual Data Entry")
            st.markdown("- Sample Data Generation")
        
        with col3:
            st.markdown("**Analysis Features**")
            st.markdown("- Comparative Analysis")
            st.markdown("- Trend Identification")
            st.markdown("- Cluster Detection")

def main():
    """Main function to run the enhanced app"""
    app = EnhancedStudentPredictorApp()
    app.run()

if __name__ == "__main__":
    main()