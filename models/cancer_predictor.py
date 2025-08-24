import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class CancerPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.cancer_types = [
            'Lung', 'Breast', 'Colorectal', 'Prostate', 'Stomach',
            'Liver', 'Cervical', 'Esophageal', 'Bladder', 'Kidney'
        ]
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize and train the cancer prediction model with synthetic medical data"""
        # Create training data based on medical literature patterns
        np.random.seed(42)
        n_samples = 10000
        
        # Generate features
        features = []
        labels = []
        
        for i in range(n_samples):
            # Demographics
            age = np.random.normal(60, 15)
            age = max(20, min(90, age))
            gender = np.random.choice([0, 1])  # 0: Male, 1: Female
            weight = np.random.normal(75, 15)
            height = np.random.normal(170, 10)
            bmi = weight / ((height/100) ** 2)
            
            # Risk factors
            smoking = np.random.choice([0, 1, 2], p=[0.6, 0.25, 0.15])  # Never, Former, Current
            alcohol = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.25, 0.05])  # None, Light, Moderate, Heavy
            family_history = np.random.choice([0, 1], p=[0.7, 0.3])
            
            # Symptoms (0-10 scale or binary)
            fatigue = np.random.poisson(2)
            pain = np.random.poisson(1.5)
            weight_loss = np.random.poisson(1)
            fever = np.random.choice([0, 1], p=[0.8, 0.2])
            night_sweats = np.random.choice([0, 1], p=[0.85, 0.15])
            appetite_loss = np.random.choice([0, 1], p=[0.75, 0.25])
            
            # Location-specific symptoms
            chest_pain = np.random.choice([0, 1], p=[0.9, 0.1])
            cough = np.random.choice([0, 1], p=[0.85, 0.15])
            swallowing = np.random.choice([0, 1], p=[0.95, 0.05])
            abdominal_pain = np.random.choice([0, 1], p=[0.9, 0.1])
            bowel_changes = np.random.choice([0, 1], p=[0.92, 0.08])
            urinary_changes = np.random.choice([0, 1], p=[0.88, 0.12])
            skin_changes = np.random.choice([0, 1], p=[0.93, 0.07])
            breast_changes = np.random.choice([0, 1], p=[0.95, 0.05])
            
            feature_vector = [
                age, gender, bmi, smoking, alcohol, family_history,
                fatigue, pain, weight_loss, fever, night_sweats, appetite_loss,
                chest_pain, cough, swallowing, abdominal_pain, bowel_changes,
                urinary_changes, skin_changes, breast_changes
            ]
            
            # Determine cancer type based on risk factors and symptoms
            cancer_prob = self._calculate_cancer_probability(feature_vector)
            
            if cancer_prob > 0.3:
                # Select cancer type based on symptoms and demographics
                cancer_type = self._select_cancer_type(feature_vector)
                labels.append(cancer_type)
            else:
                labels.append('No Cancer')
            
            features.append(feature_vector)
        
        # Convert to arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Train model
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Use ensemble of models
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y_encoded)
    
    def _calculate_cancer_probability(self, features):
        """Calculate base cancer probability from features"""
        age, gender, bmi, smoking, alcohol, family_history = features[:6]
        fatigue, pain, weight_loss, fever, night_sweats, appetite_loss = features[6:12]
        
        # Base probability increases with age
        prob = 0.1 + (age - 40) * 0.005
        
        # Risk factors
        if smoking == 2:  # Current smoker
            prob += 0.15
        elif smoking == 1:  # Former smoker
            prob += 0.08
        
        if alcohol >= 2:  # Moderate to heavy drinking
            prob += 0.05
        
        if family_history:
            prob += 0.1
        
        # Symptom severity
        symptom_score = (fatigue + pain + weight_loss) / 30.0
        prob += symptom_score * 0.2
        
        if fever or night_sweats or appetite_loss:
            prob += 0.1
        
        return min(prob, 0.9)
    
    def _select_cancer_type(self, features):
        """Select most likely cancer type based on features"""
        age, gender, bmi, smoking, alcohol, family_history = features[:6]
        symptoms = features[6:]
        
        # Age and gender specific probabilities
        if gender == 1 and age > 40:  # Female
            if features[19]:  # breast_changes
                return 'Breast'
        
        if gender == 0 and age > 50:  # Male
            if features[17]:  # urinary_changes
                return 'Prostate'
        
        # Smoking related
        if smoking >= 1:
            if features[12] or features[13]:  # chest_pain or cough
                return 'Lung'
        
        # Digestive symptoms
        if features[15] or features[16]:  # abdominal_pain or bowel_changes
            return 'Colorectal'
        
        # Default based on age and gender
        if age > 60:
            return np.random.choice(['Lung', 'Colorectal', 'Stomach'], p=[0.4, 0.3, 0.3])
        else:
            return np.random.choice(['Breast', 'Cervical', 'Kidney'], p=[0.4, 0.3, 0.3])
    
    def predict_cancer_type(self, patient_data):
        """Predict cancer type and probability"""
        # Extract features from patient data
        features = self._extract_features(patient_data)
        features_scaled = self.scaler.transform([features])
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        predicted_class = self.model.predict(features_scaled)[0]
        
        # Convert back to cancer type
        cancer_type = self.label_encoder.inverse_transform([predicted_class])[0]
        max_prob = max(probabilities)
        
        # Calculate risk factors
        risk_factors = self._calculate_risk_factors(patient_data)
        
        return {
            'cancer_type': cancer_type,
            'probability': max_prob,
            'all_probabilities': dict(zip(self.label_encoder.classes_, probabilities)),
            'risk_factors': risk_factors
        }
    
    def _extract_features(self, patient_data):
        """Extract numerical features from patient data"""
        # Demographics
        age = patient_data['age']
        gender = 1 if patient_data['gender'] == 'Female' else 0
        bmi = patient_data['weight'] / ((patient_data['height']/100) ** 2)
        
        # Risk factors
        smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
        smoking = smoking_map.get(patient_data['smoking_history'], 0)
        
        alcohol_map = {'None': 0, 'Light': 1, 'Moderate': 2, 'Heavy': 3}
        alcohol = alcohol_map.get(patient_data['alcohol_consumption'], 0)
        
        family_history = 1 if patient_data['family_history'] and 'None' not in patient_data['family_history'] else 0
        
        # Symptoms
        symptoms = patient_data['symptoms']
        fatigue = symptoms.get('fatigue', 0)
        pain = symptoms.get('pain', 0)
        weight_loss = symptoms.get('weight_loss', 0)
        fever = 1 if symptoms.get('fever', False) else 0
        night_sweats = 1 if symptoms.get('night_sweats', False) else 0
        appetite_loss = 1 if symptoms.get('loss_of_appetite', False) else 0
        
        # Location-specific symptoms
        chest_pain = 1 if symptoms.get('chest_pain', False) else 0
        cough = 1 if symptoms.get('persistent_cough', False) else 0
        swallowing = 1 if symptoms.get('difficulty_swallowing', False) else 0
        abdominal_pain = 1 if symptoms.get('abdominal_pain', False) else 0
        bowel_changes = 1 if symptoms.get('changes_in_bowel', False) else 0
        urinary_changes = 1 if symptoms.get('urinary_changes', False) else 0
        skin_changes = 1 if symptoms.get('skin_changes', False) else 0
        breast_changes = 1 if symptoms.get('breast_changes', False) else 0
        
        return [
            age, gender, bmi, smoking, alcohol, family_history,
            fatigue, pain, weight_loss, fever, night_sweats, appetite_loss,
            chest_pain, cough, swallowing, abdominal_pain, bowel_changes,
            urinary_changes, skin_changes, breast_changes
        ]
    
    def _calculate_risk_factors(self, patient_data):
        """Calculate risk factor contributions"""
        risk_factors = {}
        
        # Age risk
        age = patient_data['age']
        if age > 65:
            risk_factors['Age (>65)'] = 0.8
        elif age > 50:
            risk_factors['Age (50-65)'] = 0.5
        else:
            risk_factors['Age (<50)'] = 0.2
        
        # Smoking
        smoking = patient_data['smoking_history']
        if smoking == 'Current':
            risk_factors['Current Smoking'] = 0.9
        elif smoking == 'Former':
            risk_factors['Former Smoking'] = 0.6
        
        # Family history
        if patient_data['family_history'] and 'None' not in patient_data['family_history']:
            risk_factors['Family History'] = 0.7
        
        # Symptoms severity
        symptoms = patient_data['symptoms']
        total_symptoms = sum([
            symptoms.get('fatigue', 0) > 5,
            symptoms.get('pain', 0) > 5,
            symptoms.get('weight_loss', 0) > 5,
            symptoms.get('fever', False),
            symptoms.get('night_sweats', False),
            symptoms.get('loss_of_appetite', False)
        ])
        
        if total_symptoms >= 3:
            risk_factors['Severe Symptoms'] = 0.8
        elif total_symptoms >= 2:
            risk_factors['Moderate Symptoms'] = 0.6
        elif total_symptoms >= 1:
            risk_factors['Mild Symptoms'] = 0.3
        
        return risk_factors
