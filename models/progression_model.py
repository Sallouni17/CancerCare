import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

class ProgressionModel:
    def __init__(self):
        self.progression_models = {}
        self.symptom_models = {}
        self.scalers = {}
        self.progression_rates = {
            'Stage 0': {'median_months': 60, 'range': (36, 120)},
            'Stage I': {'median_months': 24, 'range': (18, 36)},
            'Stage II': {'median_months': 18, 'range': (12, 24)},
            'Stage III': {'median_months': 12, 'range': (8, 18)},
            'Stage IV': {'median_months': 6, 'range': (3, 12)}
        }
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize progression prediction models"""
        stages = ['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
        
        for stage in stages:
            # Generate training data for progression
            X_prog, y_prog = self._generate_progression_data(stage)
            X_symp, y_symp = self._generate_symptom_data(stage)
            
            # Train progression model
            scaler_prog = StandardScaler()
            X_prog_scaled = scaler_prog.fit_transform(X_prog)
            
            model_prog = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model_prog.fit(X_prog_scaled, y_prog)
            
            # Train symptom prediction model
            scaler_symp = StandardScaler()
            X_symp_scaled = scaler_symp.fit_transform(X_symp)
            
            model_symp = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model_symp.fit(X_symp_scaled, y_symp)
            
            self.progression_models[stage] = model_prog
            self.symptom_models[stage] = model_symp
            self.scalers[f'{stage}_prog'] = scaler_prog
            self.scalers[f'{stage}_symp'] = scaler_symp
    
    def _generate_progression_data(self, stage, n_samples=2000):
        """Generate synthetic progression data"""
        np.random.seed(42)
        
        features = []
        progression_times = []
        
        base_time = self.progression_rates[stage]['median_months']
        time_range = self.progression_rates[stage]['range']
        
        for i in range(n_samples):
            # Patient factors
            age = np.random.normal(60, 15)
            age = max(20, min(90, age))
            
            # Performance status
            performance_status = np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.3, 0.2, 0.15, 0.05])
            
            # Treatment response
            treatment_response = np.random.uniform(0, 1)
            
            # Comorbidities
            comorbidities = np.random.poisson(1.5)
            
            # Genetic factors (simplified)
            genetic_risk = np.random.uniform(0, 1)
            
            # Lifestyle factors
            smoking_status = np.random.choice([0, 1, 2], p=[0.6, 0.25, 0.15])
            exercise_level = np.random.uniform(0, 1)
            diet_quality = np.random.uniform(0, 1)
            
            # Biomarkers
            tumor_markers = np.random.lognormal(0, 1)
            
            feature_vector = [
                age, performance_status, treatment_response, comorbidities,
                genetic_risk, smoking_status, exercise_level, diet_quality,
                tumor_markers
            ]
            
            # Calculate progression time
            # Younger patients, better performance status, better treatment response = longer time
            time_modifier = 1.0
            time_modifier *= (1.0 - (age - 40) / 100)  # Age factor
            time_modifier *= (1.0 - performance_status * 0.2)  # Performance factor
            time_modifier *= (0.5 + treatment_response * 0.5)  # Treatment factor
            time_modifier *= (1.0 - comorbidities * 0.1)  # Comorbidity factor
            time_modifier *= (0.7 + diet_quality * 0.3)  # Lifestyle factor
            
            progression_time = base_time * max(0.1, time_modifier)
            progression_time = max(time_range[0], min(time_range[1], progression_time))
            
            features.append(feature_vector)
            progression_times.append(progression_time)
        
        return np.array(features), np.array(progression_times)
    
    def _generate_symptom_data(self, stage, n_samples=2000):
        """Generate synthetic symptom progression data"""
        np.random.seed(42)
        
        features = []
        symptom_scores = []
        
        base_severity = {'Stage 0': 2, 'Stage I': 3, 'Stage II': 5, 
                        'Stage III': 7, 'Stage IV': 9}[stage]
        
        for i in range(n_samples):
            # Current symptom levels
            current_pain = np.random.uniform(0, 10)
            current_fatigue = np.random.uniform(0, 10)
            current_nausea = np.random.uniform(0, 10)
            
            # Time since diagnosis
            months_since_diagnosis = np.random.exponential(6)
            
            # Treatment effects
            on_treatment = np.random.choice([0, 1], p=[0.3, 0.7])
            treatment_effectiveness = np.random.uniform(0, 1)
            
            # Patient factors
            age = np.random.normal(60, 15)
            performance_status = np.random.choice([0, 1, 2, 3, 4])
            
            feature_vector = [
                current_pain, current_fatigue, current_nausea,
                months_since_diagnosis, on_treatment, treatment_effectiveness,
                age, performance_status
            ]
            
            # Calculate future symptom severity
            severity_modifier = 1.0
            severity_modifier *= (1.0 + months_since_diagnosis * 0.1)  # Time progression
            severity_modifier *= (1.0 - treatment_effectiveness * on_treatment * 0.3)  # Treatment effect
            severity_modifier *= (1.0 + performance_status * 0.1)  # Performance status
            
            future_severity = base_severity * severity_modifier
            future_severity = max(0, min(10, future_severity))
            
            features.append(feature_vector)
            symptom_scores.append(future_severity)
        
        return np.array(features), np.array(symptom_scores)
    
    def predict_progression(self, patient_data, stage_prediction):
        """Predict disease progression timeline"""
        stage = stage_prediction['stage']
        
        if stage not in self.progression_models:
            stage = 'Stage I'  # Default
        
        # Extract features
        features = self._extract_progression_features(patient_data, stage_prediction)
        
        # Scale and predict
        features_scaled = self.scalers[f'{stage}_prog'].transform([features])
        predicted_months = self.progression_models[stage].predict(features_scaled)[0]
        
        # Generate progression timeline
        timeline = self._generate_progression_timeline(stage, predicted_months)
        
        return {
            'current_stage': stage,
            'predicted_progression_months': predicted_months,
            'timeline': timeline,
            'risk_factors': self._calculate_progression_risks(patient_data),
            'confidence_interval': (predicted_months * 0.7, predicted_months * 1.3)
        }
    
    def predict_next_stage_symptoms(self, current_stage):
        """Predict symptoms for next stage progression"""
        stage_progression = {
            'Stage 0': 'Stage I',
            'Stage I': 'Stage II',
            'Stage II': 'Stage III',
            'Stage III': 'Stage IV',
            'Stage IV': 'Stage IV'  # Terminal stage
        }
        
        next_stage = stage_progression.get(current_stage, current_stage)
        
        # Define symptom probabilities for each stage
        symptom_probabilities = {
            'Stage I': [
                ('Mild fatigue', 0.7, 'Mild'),
                ('Occasional pain', 0.6, 'Mild'),
                ('Slight weight loss', 0.5, 'Mild'),
                ('Reduced appetite', 0.4, 'Mild')
            ],
            'Stage II': [
                ('Moderate fatigue', 0.8, 'Moderate'),
                ('Persistent pain', 0.7, 'Moderate'),
                ('Noticeable weight loss', 0.6, 'Moderate'),
                ('Sleep disturbances', 0.5, 'Moderate'),
                ('Mood changes', 0.4, 'Mild')
            ],
            'Stage III': [
                ('Severe fatigue', 0.9, 'Severe'),
                ('Significant pain', 0.8, 'Severe'),
                ('Substantial weight loss', 0.8, 'Severe'),
                ('Breathing difficulties', 0.6, 'Moderate'),
                ('Nausea and vomiting', 0.7, 'Moderate'),
                ('Cognitive changes', 0.5, 'Moderate')
            ],
            'Stage IV': [
                ('Extreme fatigue', 0.95, 'Severe'),
                ('Severe pain', 0.9, 'Severe'),
                ('Significant weight loss', 0.9, 'Severe'),
                ('Severe breathing problems', 0.8, 'Severe'),
                ('Frequent nausea/vomiting', 0.8, 'Severe'),
                ('Confusion', 0.6, 'Moderate'),
                ('Swelling (edema)', 0.7, 'Moderate'),
                ('Jaundice', 0.4, 'Moderate')
            ]
        }
        
        return symptom_probabilities.get(next_stage, [])
    
    def _extract_progression_features(self, patient_data, stage_prediction):
        """Extract features for progression prediction"""
        age = patient_data['age']
        symptoms = patient_data['symptoms']
        
        # Performance status estimation
        fatigue = symptoms.get('fatigue', 0)
        pain = symptoms.get('pain', 0)
        performance_status = min(4, max(0, (fatigue + pain) / 5))
        
        # Treatment response (estimated)
        treatment_response = max(0, 1 - (fatigue + pain) / 20)
        
        # Comorbidities (estimated from patient data)
        comorbidities = 0
        if patient_data['smoking_history'] != 'Never':
            comorbidities += 1
        if patient_data['alcohol_consumption'] in ['Moderate', 'Heavy']:
            comorbidities += 1
        if age > 65:
            comorbidities += 1
        
        # Genetic risk (estimated from family history)
        genetic_risk = 0.3
        if patient_data['family_history'] and 'None' not in patient_data['family_history']:
            genetic_risk = 0.7
        
        # Lifestyle factors
        smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
        smoking_status = smoking_map.get(patient_data['smoking_history'], 0)
        
        # Estimated lifestyle factors
        exercise_level = 0.7 if fatigue < 5 else 0.3
        diet_quality = 0.6  # Baseline assumption
        
        # Tumor markers (estimated)
        tumor_markers = (pain + fatigue) / 10.0
        
        return [
            age, performance_status, treatment_response, comorbidities,
            genetic_risk, smoking_status, exercise_level, diet_quality,
            tumor_markers
        ]
    
    def _generate_progression_timeline(self, stage, predicted_months):
        """Generate detailed progression timeline"""
        timeline = []
        current_date = datetime.now()
        
        # Define progression milestones
        milestones = {
            'Stage 0': [
                (0.2, 'Regular monitoring begins'),
                (0.5, 'First follow-up assessment'),
                (0.8, 'Continued surveillance'),
                (1.0, 'Re-evaluation for progression')
            ],
            'Stage I': [
                (0.1, 'Treatment initiation'),
                (0.3, 'Initial treatment response evaluation'),
                (0.6, 'Mid-treatment assessment'),
                (0.9, 'Treatment completion evaluation'),
                (1.0, 'Possible progression to Stage II')
            ],
            'Stage II': [
                (0.1, 'Aggressive treatment begins'),
                (0.25, 'First treatment cycle completion'),
                (0.5, 'Mid-treatment imaging'),
                (0.75, 'Treatment response assessment'),
                (1.0, 'Possible progression to Stage III')
            ],
            'Stage III': [
                (0.1, 'Multi-modal treatment initiation'),
                (0.3, 'Symptom management intensification'),
                (0.6, 'Treatment modification as needed'),
                (0.8, 'Palliative care consultation'),
                (1.0, 'Possible progression to Stage IV')
            ],
            'Stage IV': [
                (0.2, 'Palliative care focus'),
                (0.4, 'Symptom control optimization'),
                (0.6, 'Quality of life assessment'),
                (0.8, 'End-of-life planning'),
                (1.0, 'Terminal care')
            ]
        }
        
        stage_milestones = milestones.get(stage, milestones['Stage I'])
        
        for fraction, description in stage_milestones:
            milestone_date = current_date + timedelta(days=predicted_months * 30 * fraction)
            timeline.append({
                'date': milestone_date.strftime('%Y-%m-%d'),
                'months_from_now': predicted_months * fraction,
                'description': description,
                'likelihood': max(0.5, 1.0 - fraction * 0.3)  # Decreasing certainty over time
            })
        
        return timeline
    
    def _calculate_progression_risks(self, patient_data):
        """Calculate risk factors for progression"""
        risks = {}
        
        # Age risk
        age = patient_data['age']
        if age > 70:
            risks['Advanced Age'] = 0.8
        elif age > 60:
            risks['Older Age'] = 0.6
        
        # Smoking
        smoking = patient_data['smoking_history']
        if smoking == 'Current':
            risks['Current Smoking'] = 0.9
        elif smoking == 'Former':
            risks['Smoking History'] = 0.6
        
        # Performance status
        symptoms = patient_data['symptoms']
        fatigue = symptoms.get('fatigue', 0)
        pain = symptoms.get('pain', 0)
        
        if fatigue + pain > 15:
            risks['Poor Performance Status'] = 0.8
        elif fatigue + pain > 10:
            risks['Moderate Performance Status'] = 0.6
        
        # Weight loss
        weight_loss = symptoms.get('weight_loss', 0)
        if weight_loss > 10:
            risks['Significant Weight Loss'] = 0.9
        elif weight_loss > 5:
            risks['Moderate Weight Loss'] = 0.6
        
        # Comorbidities
        if patient_data['alcohol_consumption'] in ['Moderate', 'Heavy']:
            risks['Alcohol Use'] = 0.5
        
        return risks
