import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from models.cancer_predictor import CancerPredictor
from models.staging_model import StagingModel
from models.progression_model import ProgressionModel
from data.treatment_database import TreatmentDatabase
from data.nutrition_recommendations import NutritionRecommendations
from utils.data_validation import DataValidator
from utils.visualization import VisualizationHelper

import streamlit as st
from datetime import datetime
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
import plotly.graph_objects as go
import plotly.io as pio

# Configure page
st.set_page_config(
    page_title="Medical AI Cancer Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize models and data
@st.cache_resource
def load_models():
    cancer_predictor = CancerPredictor()
    staging_model = StagingModel()
    progression_model = ProgressionModel()
    treatment_db = TreatmentDatabase()
    nutrition_db = NutritionRecommendations()
    validator = DataValidator()
    viz_helper = VisualizationHelper()
    
    return cancer_predictor, staging_model, progression_model, treatment_db, nutrition_db, validator, viz_helper

def main():
    st.title("🏥 Medical AI Cancer Diagnosis & Management System")
    st.markdown("---")
    
    # Load models
    cancer_predictor, staging_model, progression_model, treatment_db, nutrition_db, validator, viz_helper = load_models()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Module",
        ["Patient Input", "Diagnosis Results", "Treatment Planning", "Lifestyle Management", "Reports"]
    )
    
    if page == "Patient Input":
        patient_input_page(validator)
    elif page == "Diagnosis Results":
        diagnosis_results_page(cancer_predictor, staging_model, progression_model, viz_helper)
    elif page == "Treatment Planning":
        treatment_planning_page(treatment_db, viz_helper)
    elif page == "Lifestyle Management":
        lifestyle_management_page(nutrition_db)
    elif page == "Reports":
        reports_page(viz_helper)

def patient_input_page(validator):
    st.header("📝 Patient Information & Symptoms")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Demographics")
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        
        st.subheader("Medical History")
        family_history = st.multiselect(
            "Family History of Cancer",
            ["Breast", "Lung", "Colon", "Prostate", "Ovarian", "Pancreatic", "Skin", "None"]
        )
        smoking_history = st.selectbox("Smoking History", ["Never", "Former", "Current"])
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Light", "Moderate", "Heavy"])
    
    with col2:
        st.subheader("Current Symptoms")
        symptoms = {}
        
        # Physical symptoms
        symptoms['fatigue'] = st.slider("Fatigue Level (0-10)", 0, 10, 0)
        symptoms['pain'] = st.slider("Pain Level (0-10)", 0, 10, 0)
        symptoms['weight_loss'] = st.slider("Unexplained Weight Loss (kg in last 6 months)", 0, 30, 0)
        symptoms['fever'] = st.checkbox("Persistent Fever")
        symptoms['night_sweats'] = st.checkbox("Night Sweats")
        symptoms['loss_of_appetite'] = st.checkbox("Loss of Appetite")
        
        # Location-specific symptoms
        st.subheader("Location-Specific Symptoms")
        symptoms['chest_pain'] = st.checkbox("Chest Pain")
        symptoms['persistent_cough'] = st.checkbox("Persistent Cough")
        symptoms['difficulty_swallowing'] = st.checkbox("Difficulty Swallowing")
        symptoms['abdominal_pain'] = st.checkbox("Abdominal Pain")
        symptoms['changes_in_bowel'] = st.checkbox("Changes in Bowel Habits")
        symptoms['urinary_changes'] = st.checkbox("Urinary Changes")
        symptoms['skin_changes'] = st.checkbox("Skin Changes/Lumps")
        symptoms['breast_changes'] = st.checkbox("Breast Changes")
    
    # Store patient data in session state
    if st.button("Save Patient Data", type="primary"):
        patient_data = {
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'family_history': family_history,
            'smoking_history': smoking_history,
            'alcohol_consumption': alcohol_consumption,
            'symptoms': symptoms
        }
        
        # Validate data
        validation_result = validator.validate_patient_data(patient_data)
        
        if validation_result['is_valid']:
            st.session_state['patient_data'] = patient_data
            st.success("✅ Patient data saved successfully!")
            st.info("Navigate to 'Diagnosis Results' to see AI predictions.")
        else:
            st.error(f"❌ Data validation failed: {validation_result['error']}")

def diagnosis_results_page(cancer_predictor, staging_model, progression_model, viz_helper):
    st.header("🔬 AI Diagnosis Results")
    
    if 'patient_data' not in st.session_state:
        st.warning("⚠️ Please complete patient input first.")
        return
    
    patient_data = st.session_state['patient_data']
    
    # Cancer type prediction
    st.subheader("Cancer Type Prediction")
    cancer_prediction = cancer_predictor.predict_cancer_type(patient_data)
    
    col1, col2 = st.columns(2)
    with col1:
        # Display prediction results
        if cancer_prediction['probability'] > 0.6:
            st.error(f"⚠️ High Risk Detected: {cancer_prediction['cancer_type']}")
            st.write(f"**Confidence:** {cancer_prediction['probability']:.1%}")
        elif cancer_prediction['probability'] > 0.3:
            st.warning(f"⚠️ Moderate Risk: {cancer_prediction['cancer_type']}")
            st.write(f"**Confidence:** {cancer_prediction['probability']:.1%}")
        else:
            st.success("✅ Low Risk Detected")
            st.write(f"**Confidence:** {(1-cancer_prediction['probability']):.1%}")
    
    with col2:
        # Risk factors chart
        fig_risk = viz_helper.create_risk_factors_chart(cancer_prediction['risk_factors'])
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Stage prediction if cancer risk is significant
    if cancer_prediction['probability'] > 0.3:
        st.subheader("Cancer Stage Assessment")
        stage_prediction = staging_model.predict_stage(patient_data, cancer_prediction['cancer_type'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Stage", stage_prediction['stage'])
            st.metric("TNM Classification", stage_prediction['tnm'])
            st.metric("5-Year Survival Rate", f"{stage_prediction['survival_rate']:.1%}")
        
        with col2:
            fig_stage = viz_helper.create_staging_chart(stage_prediction)
            st.plotly_chart(fig_stage, use_container_width=True)
        
        # Progression prediction
        st.subheader("Disease Progression Timeline")
        progression_data = progression_model.predict_progression(patient_data, stage_prediction)
        
        fig_timeline = viz_helper.create_progression_timeline(progression_data)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Next stage symptoms
        st.subheader("Anticipated Symptoms & Severity")
        next_symptoms = progression_model.predict_next_stage_symptoms(stage_prediction['stage'])
        
        symptoms_df = pd.DataFrame([
            {"Symptom": symptom, "Likelihood": f"{likelihood:.1%}", "Severity": severity}
            for symptom, likelihood, severity in next_symptoms
        ])
        st.dataframe(symptoms_df, use_container_width=True)
        
        # Store diagnosis results
        st.session_state['diagnosis_results'] = {
            'cancer_prediction': cancer_prediction,
            'stage_prediction': stage_prediction,
            'progression_data': progression_data,
            'next_symptoms': next_symptoms
        }

def treatment_planning_page(treatment_db, viz_helper):
    st.header("💊 Treatment Planning & Scheduling")
    
    if 'diagnosis_results' not in st.session_state:
        st.warning("⚠️ Please complete diagnosis first.")
        return
    
    diagnosis = st.session_state['diagnosis_results']
    cancer_type = diagnosis['cancer_prediction']['cancer_type']
    stage = diagnosis['stage_prediction']['stage']
    
    # Treatment recommendations
    st.subheader("Recommended Treatment Plan")
    treatment_plan = treatment_db.get_treatment_plan(cancer_type, stage)
    
    if treatment_plan:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Chemotherapy schedule
            st.subheader("Chemotherapy Schedule")
            chemo_schedule = treatment_plan['chemotherapy']
            
            schedule_data = []
            start_date = datetime.now()
            
            for i, cycle in enumerate(chemo_schedule['cycles']):
                cycle_start = start_date + timedelta(weeks=i * cycle['interval_weeks'])
                schedule_data.append({
                    'Cycle': i + 1,
                    'Date': cycle_start.strftime('%Y-%m-%d'),
                    'Drugs': ', '.join(cycle['drugs']),
                    'Duration': f"{cycle['duration_days']} days",
                    'Severity': cycle['expected_severity']
                })
            
            schedule_df = pd.DataFrame(schedule_data)
            st.dataframe(schedule_df, use_container_width=True)
        
        with col2:
            st.subheader("Treatment Overview")
            st.metric("Total Cycles", len(chemo_schedule['cycles']))
            st.metric("Treatment Duration", f"{chemo_schedule['total_weeks']} weeks")
            st.metric("Success Rate", f"{treatment_plan['success_rate']:.1%}")
        
        # Treatment timeline visualization
        fig_treatment = viz_helper.create_treatment_timeline(schedule_data)
        st.plotly_chart(fig_treatment, use_container_width=True)
        
        # Side effects and monitoring
        st.subheader("Expected Side Effects & Monitoring")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Common Side Effects:**")
            for effect in treatment_plan['side_effects']:
                st.write(f"• {effect}")
        
        with col2:
            st.write("**Monitoring Schedule:**")
            for monitor in treatment_plan['monitoring']:
                st.write(f"• {monitor}")
        
        # Store treatment plan
        st.session_state['treatment_plan'] = {
            'plan': treatment_plan,
            'schedule': schedule_data
        }
    else:
        st.error("❌ No treatment plan available for the predicted cancer type and stage.")

def lifestyle_management_page(nutrition_db):
    st.header("🥗 Lifestyle & Nutrition Management")
    
    if 'diagnosis_results' not in st.session_state:
        st.warning("⚠️ Please complete diagnosis first.")
        return
    
    diagnosis = st.session_state['diagnosis_results']
    treatment_plan = st.session_state.get('treatment_plan', {})
    
    cancer_type = diagnosis['cancer_prediction']['cancer_type']
    stage = diagnosis['stage_prediction']['stage']
    
    # Nutrition recommendations
    st.subheader("Personalized Nutrition Plan")
    nutrition_plan = nutrition_db.get_nutrition_plan(cancer_type, stage)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Recommended Foods:**")
        for food in nutrition_plan['recommended_foods']:
            st.write(f"✅ {food}")
    
    with col2:
        st.write("**Foods to Avoid:**")
        for food in nutrition_plan['avoid_foods']:
            st.write(f"❌ {food}")
    
    with col3:
        st.write("**Nutritional Goals:**")
        for goal in nutrition_plan['nutritional_goals']:
            st.write(f"🎯 {goal}")
    
    # Weekly meal planning
    st.subheader("Weekly Meal Plan")
    meal_plan = nutrition_db.generate_weekly_meal_plan(nutrition_plan)
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    meals = ['Breakfast', 'Lunch', 'Dinner', 'Snacks']
    
    meal_df = pd.DataFrame(meal_plan, index=meals, columns=days)
    st.dataframe(meal_df, use_container_width=True)
    
    # Activity restrictions and recommendations
    st.subheader("Activity & Lifestyle Recommendations")
    lifestyle_rec = nutrition_db.get_lifestyle_recommendations(cancer_type, stage)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Recommended Activities:**")
        for activity in lifestyle_rec['recommended_activities']:
            st.write(f"✅ {activity}")
        
        st.write("**Restrictions:**")
        for restriction in lifestyle_rec['restrictions']:
            st.write(f"⚠️ {restriction}")
    
    with col2:
        st.write("**Sleep & Rest:**")
        for sleep_tip in lifestyle_rec['sleep_recommendations']:
            st.write(f"💤 {sleep_tip}")
        
        st.write("**Stress Management:**")
        for stress_tip in lifestyle_rec['stress_management']:
            st.write(f"🧘 {stress_tip}")


def create_pdf_report(patient_data, diagnosis, treatment_plan):
    """Generate a comprehensive PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.darkblue,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.darkblue,
        spaceBefore=20,
        spaceAfter=10
    )
    
    # Build the document content
    story = []
    
    # Title
    story.append(Paragraph("📊 Comprehensive Medical Report", title_style))
    story.append(Spacer(1, 30))
    
    # Patient Information
    story.append(Paragraph("Patient Information", heading_style))
    patient_table_data = [
        ['Patient ID:', patient_data.get('patient_id', 'N/A')],
        ['Age:', f"{patient_data.get('age', 'N/A')} years"],
        ['Gender:', patient_data.get('gender', 'N/A')],
        ['Report Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ]
    
    patient_table = Table(patient_table_data, colWidths=[2*inch, 3*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Diagnosis Results
    story.append(Paragraph("Diagnosis Results", heading_style))
    
    cancer_prob = diagnosis.get('cancer_prediction', {}).get('probability', 0)
    stage = diagnosis.get('stage_prediction', {}).get('stage', 'N/A')
    survival_rate = diagnosis.get('stage_prediction', {}).get('survival_rate', 0)
    
    diagnosis_table_data = [
        ['Cancer Risk Probability:', f"{cancer_prob:.1%}"],
        ['Predicted Stage:', str(stage)],
        ['Survival Rate:', f"{survival_rate:.1%}"],
        ['Risk Level:', 'High' if cancer_prob > 0.7 else 'Medium' if cancer_prob > 0.3 else 'Low']
    ]
    
    diagnosis_table = Table(diagnosis_table_data, colWidths=[2.5*inch, 2.5*inch])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightcoral),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.lightyellow),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(diagnosis_table)
    story.append(Spacer(1, 20))
    
    # Treatment Plan (if available)
    if treatment_plan:
        story.append(Paragraph("Treatment Plan", heading_style))
        treatment_text = f"""
        Treatment Type: {treatment_plan.get('treatment_type', 'N/A')}<br/>
        Duration: {treatment_plan.get('duration', 'N/A')}<br/>
        Expected Effectiveness: {treatment_plan.get('effectiveness', 'N/A')}<br/>
        Side Effects: {treatment_plan.get('side_effects', 'N/A')}<br/>
        Follow-up Schedule: {treatment_plan.get('follow_up', 'N/A')}
        """
        story.append(Paragraph(treatment_text, styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    recommendations = f"""
    Based on the analysis results:<br/><br/>
    
    • Regular monitoring and follow-up appointments are recommended<br/>
    • Lifestyle modifications may help improve outcomes<br/>
    • Adherence to prescribed treatment plan is crucial<br/>
    • Report any unusual symptoms immediately<br/>
    • Consider second opinion if risk levels are high<br/><br/>
    
    <b>Disclaimer:</b> This report is generated by an AI system for educational/demonstration purposes only. 
    Always consult with qualified healthcare professionals for actual medical decisions.
    """
    story.append(Paragraph(recommendations, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def reports_page(viz_helper):
    st.header("📊 Comprehensive Medical Report")
    
    if 'diagnosis_results' not in st.session_state:
        st.warning("⚠️ Please complete diagnosis first.")
        return
    
    diagnosis = st.session_state['diagnosis_results']
    patient_data = st.session_state['patient_data']
    treatment_plan = st.session_state.get('treatment_plan', {})
    
    # Generate comprehensive report
    st.subheader("Patient Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age", f"{patient_data['age']} years")
    with col2:
        st.metric("Cancer Risk", f"{diagnosis['cancer_prediction']['probability']:.1%}")
    with col3:
        st.metric("Predicted Stage", diagnosis['stage_prediction']['stage'])
    with col4:
        st.metric("Survival Rate", f"{diagnosis['stage_prediction']['survival_rate']:.1%}")
    
    # Risk assessment over time
    st.subheader("Risk Assessment Timeline")
    fig_risk_timeline = viz_helper.create_comprehensive_risk_timeline(diagnosis, patient_data)
    st.plotly_chart(fig_risk_timeline, use_container_width=True)
    
    # Treatment effectiveness prediction
    if treatment_plan:
        st.subheader("Treatment Effectiveness Prediction")
        fig_effectiveness = viz_helper.create_treatment_effectiveness_chart(treatment_plan)
        st.plotly_chart(fig_effectiveness, use_container_width=True)
    
    # Downloadable report section
    st.subheader("Download Report")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # PDF Download
        if st.button("📄 Generate & Download PDF Report", type="primary"):
            try:
                with st.spinner("Generating PDF report..."):
                    # Generate PDF
                    pdf_buffer = create_pdf_report(patient_data, diagnosis, treatment_plan)
                    
                    # Create download button
                    st.download_button(
                        label="💾 Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"medical_report_{patient_data.get('patient_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        type="secondary"
                    )
                    
                st.success("✅ PDF report generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating PDF: {str(e)}")
                st.info("💡 Make sure reportlab is installed: pip install reportlab")
    
    with col2:
        # JSON Download (backup option)
        report_data = {
            'patient_data': patient_data,
            'diagnosis': diagnosis,
            'treatment_plan': treatment_plan,
            'generated_date': datetime.now().isoformat()
        }
        
        st.download_button(
            label="📋 Download JSON Data",
            data=json.dumps(report_data, indent=2),
            file_name=f"medical_data_{patient_data.get('patient_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Preview section
    with st.expander("🔍 Preview Report Data"):
        st.json(report_data)
    
    # Additional export options
    st.subheader("Additional Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export for data analysis
        if st.button("📊 Export to CSV"):
            import pandas as pd
            
            # Flatten the data for CSV
            csv_data = {
                'patient_id': [patient_data.get('patient_id', 'N/A')],
                'age': [patient_data.get('age', 'N/A')],
                'cancer_probability': [diagnosis['cancer_prediction']['probability']],
                'predicted_stage': [diagnosis['stage_prediction']['stage']],
                'survival_rate': [diagnosis['stage_prediction']['survival_rate']],
                'report_date': [datetime.now().isoformat()]
            }
            
            df = pd.DataFrame(csv_data)
            csv_string = df.to_csv(index=False)
            
            st.download_button(
                label="💾 Download CSV",
                data=csv_string,
                file_name=f"medical_summary_{patient_data.get('patient_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        # Print-friendly HTML
        if st.button("🖨️ Generate Print Version"):
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Medical Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 20px; }}
                    .section {{ margin: 30px 0; }}
                    .metric {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
                    .warning {{ background: #ffeaa7; padding: 10px; border: 1px solid #fdcb6e; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>📊 Comprehensive Medical Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Patient Information</h2>
                    <div class="metric">Age: {patient_data.get('age', 'N/A')} years</div>
                    <div class="metric">Patient ID: {patient_data.get('patient_id', 'N/A')}</div>
                </div>
                
                <div class="section">
                    <h2>Diagnosis Results</h2>
                    <div class="metric">Cancer Risk: {diagnosis['cancer_prediction']['probability']:.1%}</div>
                    <div class="metric">Predicted Stage: {diagnosis['stage_prediction']['stage']}</div>
                    <div class="metric">Survival Rate: {diagnosis['stage_prediction']['survival_rate']:.1%}</div>
                </div>
                
                <div class="warning">
                    <strong>Disclaimer:</strong> This report is generated for educational/demonstration purposes only. 
                    Always consult with qualified healthcare professionals for actual medical decisions.
                </div>
            </body>
            </html>
            """
            
            st.download_button(
                label="💾 Download HTML Report",
                data=html_content,
                file_name=f"medical_report_{patient_data.get('patient_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
    
    with col3:
        # Email-ready summary
        if st.button("📧 Email Summary"):
            email_content = f"""
                Subject: Medical Report Summary - {patient_data.get('patient_id', 'Unknown Patient')}

                Dear Healthcare Provider,

                Please find below a summary of the medical analysis report:

                Patient Details:
                - Patient ID: {patient_data.get('patient_id', 'N/A')}
                - Age: {patient_data.get('age', 'N/A')} years
                - Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                Analysis Results:
                - Cancer Risk Probability: {diagnosis['cancer_prediction']['probability']:.1%}
                - Predicted Stage: {diagnosis['stage_prediction']['stage']}
                - Survival Rate: {diagnosis['stage_prediction']['survival_rate']:.1%}

                Recommendations:
                - Regular monitoring recommended
                - Follow prescribed treatment plan
                - Report unusual symptoms immediately

                DISCLAIMER: This report is generated by an AI system for educational purposes only. 
                Always consult with qualified healthcare professionals for actual medical decisions.

                Best regards,
                Medical Analysis System
                            """
            
            st.download_button(
                label="💾 Download Email Template",
                data=email_content,
                file_name=f"email_summary_{patient_data.get('patient_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
