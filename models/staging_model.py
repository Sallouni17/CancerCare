import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class StagingModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.stage_info = {
            'Stage 0': {'survival_rate': 0.99, 'description': 'In situ - abnormal cells present but not spread'},
            'Stage I': {'survival_rate': 0.95, 'description': 'Early stage - small tumor, no lymph nodes'},
            'Stage II': {'survival_rate': 0.85, 'description': 'Moderate - larger tumor or nearby lymph nodes'},
            'Stage III': {'survival_rate': 0.65, 'description': 'Advanced - spread to multiple lymph nodes'},
            'Stage IV': {'survival_rate': 0.25, 'description': 'Metastatic - spread to distant organs'}
        }
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize staging models for different cancer types"""
        cancer_types = ['Lung', 'Breast', 'Colorectal', 'Prostate', 'Stomach']
        
        for cancer_type in cancer_types:
            # Generate training data for each cancer type
            X, y = self._generate_staging_data(cancer_type)
            
            # Train model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            self.models[cancer_type] = model
            self.scalers[cancer_type] = scaler
    
    def _generate_staging_data(self, cancer_type, n_samples=5000):
        """Generate synthetic staging data based on medical patterns"""
        np.random.seed(42)
        
        features = []
        stages = []
        
        for i in range(n_samples):
            # Patient characteristics
            age = np.random.normal(60, 15)
            age = max(20, min(90, age))
            
            # Tumor characteristics
            tumor_size = np.random.exponential(3)  # cm
            lymph_nodes_affected = np.random.poisson(2)
            
            # Symptoms severity
            symptom_severity = np.random.uniform(0, 10)
            pain_level = np.random.uniform(0, 10)
            weight_loss = np.random.exponential(2)
            
            # Performance status (0-4 scale, 0 being best)
            performance_status = np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.3, 0.2, 0.15, 0.05])
            
            # Biomarkers (normalized values)
            cea_level = np.random.lognormal(1, 0.5)  # Carcinoembryonic antigen
            ca125_level = np.random.lognormal(2, 0.7)  # Cancer antigen 125
            
            # Blood markers
            hemoglobin = np.random.normal(13, 2)
            albumin = np.random.normal(4, 0.5)
            
            feature_vector = [
                age, tumor_size, lymph_nodes_affected, symptom_severity,
                pain_level, weight_loss, performance_status, cea_level,
                ca125_level, hemoglobin, albumin
            ]
            
            # Determine stage based on clinical criteria
            stage = self._determine_stage(tumor_size, lymph_nodes_affected, 
                                        performance_status, cancer_type)
            
            features.append(feature_vector)
            stages.append(stage)
        
        return np.array(features), np.array(stages)
    
    def _determine_stage(self, tumor_size, lymph_nodes_affected, performance_status, cancer_type):
        """Determine cancer stage based on clinical criteria"""
        # Simplified staging based on TNM system
        
        # T (Tumor size)
        if tumor_size < 2:
            t_stage = 1
        elif tumor_size < 5:
            t_stage = 2
        elif tumor_size < 7:
            t_stage = 3
        else:
            t_stage = 4
        
        # N (Lymph nodes)
        if lymph_nodes_affected == 0:
            n_stage = 0
        elif lymph_nodes_affected < 3:
            n_stage = 1
        elif lymph_nodes_affected < 9:
            n_stage = 2
        else:
            n_stage = 3
        
        # M (Metastasis) - simplified based on performance status
        m_stage = 1 if performance_status >= 3 else 0
        
        # Overall stage determination
        if m_stage == 1:
            return 'Stage IV'
        elif t_stage >= 3 or n_stage >= 2:
            return 'Stage III'
        elif t_stage == 2 or n_stage == 1:
            return 'Stage II'
        elif t_stage == 1 and n_stage == 0:
            return 'Stage I'
        else:
            return 'Stage 0'
    
    def predict_stage(self, patient_data, cancer_type):
        """Predict cancer stage"""
        if cancer_type not in self.models:
            # Use general model for unknown cancer types
            cancer_type = 'Lung'  # Default to lung cancer model
        
        # Extract features
        features = self._extract_staging_features(patient_data, cancer_type)
        
        # Scale features
        features_scaled = self.scalers[cancer_type].transform([features])
        
        # Predict stage
        predicted_stage = self.models[cancer_type].predict(features_scaled)[0]
        probabilities = self.models[cancer_type].predict_proba(features_scaled)[0]
        
        # Get stage classes
        stage_classes = self.models[cancer_type].classes_
        stage_probs = dict(zip(stage_classes, probabilities))
        
        # Generate TNM classification
        tnm = self._generate_tnm(predicted_stage, patient_data)
        
        # Get survival rate
        survival_rate = self.stage_info[predicted_stage]['survival_rate']
        
        return {
            'stage': predicted_stage,
            'confidence': max(probabilities),
            'all_probabilities': stage_probs,
            'tnm': tnm,
            'survival_rate': survival_rate,
            'description': self.stage_info[predicted_stage]['description']
        }
    
    def _extract_staging_features(self, patient_data, cancer_type):
        """Extract features for staging prediction"""
        symptoms = patient_data['symptoms']
        age = patient_data['age']
        
        # Estimate tumor characteristics based on symptoms
        pain_level = symptoms.get('pain', 0)
        weight_loss = symptoms.get('weight_loss', 0)
        fatigue = symptoms.get('fatigue', 0)
        
        # Estimate tumor size based on symptom severity
        tumor_size = (pain_level + weight_loss + fatigue) / 3.0
        
        # Estimate lymph node involvement
        lymph_nodes = pain_level * 0.3
        
        # Calculate performance status
        performance_status = min(4, max(0, (fatigue + pain_level) / 5))
        
        # Simulated biomarkers based on symptoms
        cea_level = 1.0 + weight_loss * 0.2
        ca125_level = 2.0 + fatigue * 0.1
        
        # Estimated blood markers
        hemoglobin = 13 - (fatigue * 0.3)
        albumin = 4.0 - (weight_loss * 0.1)
        
        return [
            age, tumor_size, lymph_nodes, (pain_level + fatigue) / 2,
            pain_level, weight_loss, performance_status, cea_level,
            ca125_level, hemoglobin, albumin
        ]
    
    def _generate_tnm(self, stage, patient_data):
        """Generate TNM classification based on stage"""
        tnm_mapping = {
            'Stage 0': 'Tis N0 M0',
            'Stage I': 'T1 N0 M0',
            'Stage II': 'T2 N0-1 M0',
            'Stage III': 'T3 N1-2 M0',
            'Stage IV': 'T4 N3 M1'
        }
        
        return tnm_mapping.get(stage, 'Unknown TNM')
    
    def get_stage_characteristics(self, stage):
        """Get detailed characteristics of a cancer stage"""
        characteristics = {
            'Stage 0': {
                'size': 'Abnormal cells only',
                'spread': 'No spread to nearby tissue',
                'lymph_nodes': 'No lymph node involvement',
                'metastasis': 'No distant spread',
                'treatment': 'Surgery, monitoring'
            },
            'Stage I': {
                'size': 'Small tumor (< 2cm)',
                'spread': 'Limited to organ of origin',
                'lymph_nodes': 'No lymph node involvement',
                'metastasis': 'No distant spread',
                'treatment': 'Surgery, possible adjuvant therapy'
            },
            'Stage II': {
                'size': 'Medium tumor (2-5cm) or nearby spread',
                'spread': 'May involve nearby tissues',
                'lymph_nodes': 'Limited lymph node involvement',
                'metastasis': 'No distant spread',
                'treatment': 'Surgery, chemotherapy, radiation'
            },
            'Stage III': {
                'size': 'Large tumor (>5cm) or extensive local spread',
                'spread': 'Significant local/regional spread',
                'lymph_nodes': 'Multiple lymph nodes involved',
                'metastasis': 'No distant spread',
                'treatment': 'Combination therapy, aggressive treatment'
            },
            'Stage IV': {
                'size': 'Any size',
                'spread': 'Widespread',
                'lymph_nodes': 'Extensive involvement possible',
                'metastasis': 'Distant organ involvement',
                'treatment': 'Palliative care, systemic therapy'
            }
        }
        
        return characteristics.get(stage, {})
