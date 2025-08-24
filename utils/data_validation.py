import re
from datetime import datetime

class DataValidator:
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self):
        """Initialize data validation rules"""
        return {
            'age': {'min': 1, 'max': 120},
            'weight': {'min': 20.0, 'max': 300.0},
            'height': {'min': 100.0, 'max': 250.0},
            'symptom_scales': {'min': 0, 'max': 10},
            'required_fields': ['age', 'gender', 'weight', 'height', 'symptoms'],
            'valid_genders': ['Male', 'Female', 'Other'],
            'valid_smoking_status': ['Never', 'Former', 'Current'],
            'valid_alcohol_levels': ['None', 'Light', 'Moderate', 'Heavy']
        }
    
    def validate_patient_data(self, patient_data):
        """Comprehensive patient data validation"""
        errors = []
        warnings = []
        
        try:
            # Check required fields
            for field in self.validation_rules['required_fields']:
                if field not in patient_data or patient_data[field] is None:
                    errors.append(f"Missing required field: {field}")
            
            if errors:
                return {'is_valid': False, 'error': '; '.join(errors), 'warnings': warnings}
            
            # Validate age
            age = patient_data.get('age')
            if not self._validate_numeric_range(age, 'age'):
                errors.append(f"Age must be between {self.validation_rules['age']['min']} and {self.validation_rules['age']['max']}")
            
            # Age-specific warnings
            if age and age < 18:
                warnings.append("Pediatric cases require specialized protocols")
            elif age and age > 90:
                warnings.append("Advanced age may affect treatment options")
            
            # Validate gender
            gender = patient_data.get('gender')
            if gender not in self.validation_rules['valid_genders']:
                errors.append(f"Gender must be one of: {', '.join(self.validation_rules['valid_genders'])}")
            
            # Validate physical measurements
            weight = patient_data.get('weight')
            height = patient_data.get('height')
            
            if not self._validate_numeric_range(weight, 'weight'):
                errors.append(f"Weight must be between {self.validation_rules['weight']['min']} and {self.validation_rules['weight']['max']} kg")
            
            if not self._validate_numeric_range(height, 'height'):
                errors.append(f"Height must be between {self.validation_rules['height']['min']} and {self.validation_rules['height']['max']} cm")
            
            # Calculate and validate BMI
            if weight and height:
                bmi = weight / ((height/100) ** 2)
                if bmi < 16:
                    warnings.append("Severely underweight BMI detected - nutritional support may be critical")
                elif bmi > 40:
                    warnings.append("Severely obese BMI detected - may affect treatment dosing")
            
            # Validate lifestyle factors
            smoking = patient_data.get('smoking_history')
            if smoking and smoking not in self.validation_rules['valid_smoking_status']:
                errors.append(f"Smoking history must be one of: {', '.join(self.validation_rules['valid_smoking_status'])}")
            
            alcohol = patient_data.get('alcohol_consumption')
            if alcohol and alcohol not in self.validation_rules['valid_alcohol_levels']:
                errors.append(f"Alcohol consumption must be one of: {', '.join(self.validation_rules['valid_alcohol_levels'])}")
            
            # Validate symptoms
            symptoms = patient_data.get('symptoms', {})
            symptom_errors = self._validate_symptoms(symptoms)
            errors.extend(symptom_errors)
            
            # Check for concerning symptom combinations
            symptom_warnings = self._check_symptom_patterns(symptoms, age, gender)
            warnings.extend(symptom_warnings)
            
            # Validate family history
            family_history = patient_data.get('family_history', [])
            if family_history and not isinstance(family_history, list):
                errors.append("Family history must be a list")
            
            # Medical logic validation
            medical_warnings = self._validate_medical_logic(patient_data)
            warnings.extend(medical_warnings)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return {
            'is_valid': len(errors) == 0,
            'error': '; '.join(errors) if errors else None,
            'warnings': warnings
        }
    
    def _validate_numeric_range(self, value, field_type):
        """Validate numeric values within acceptable ranges"""
        if value is None:
            return False
        
        try:
            value = float(value)
            rules = self.validation_rules.get(field_type)
            if rules:
                return rules['min'] <= value <= rules['max']
        except (ValueError, TypeError):
            return False
        
        return True
    
    def _validate_symptoms(self, symptoms):
        """Validate symptom data"""
        errors = []
        
        if not isinstance(symptoms, dict):
            errors.append("Symptoms must be provided as a dictionary")
            return errors
        
        # Validate symptom scales (0-10)
        scale_symptoms = ['fatigue', 'pain', 'weight_loss']
        for symptom in scale_symptoms:
            if symptom in symptoms:
                value = symptoms[symptom]
                if not self._validate_numeric_range(value, 'symptom_scales'):
                    errors.append(f"{symptom.replace('_', ' ').title()} must be between 0 and 10")
        
        # Validate boolean symptoms
        boolean_symptoms = [
            'fever', 'night_sweats', 'loss_of_appetite', 'chest_pain',
            'persistent_cough', 'difficulty_swallowing', 'abdominal_pain',
            'changes_in_bowel', 'urinary_changes', 'skin_changes', 'breast_changes'
        ]
        
        for symptom in boolean_symptoms:
            if symptom in symptoms:
                value = symptoms[symptom]
                if not isinstance(value, bool):
                    errors.append(f"{symptom.replace('_', ' ').title()} must be true or false")
        
        return errors
    
    def _check_symptom_patterns(self, symptoms, age, gender):
        """Check for concerning symptom patterns"""
        warnings = []
        
        # Check for severe symptom combinations
        high_severity_symptoms = 0
        if symptoms.get('fatigue', 0) > 7:
            high_severity_symptoms += 1
        if symptoms.get('pain', 0) > 7:
            high_severity_symptoms += 1
        if symptoms.get('weight_loss', 0) > 10:
            high_severity_symptoms += 1
        
        if high_severity_symptoms >= 2:
            warnings.append("Multiple severe symptoms detected - urgent medical evaluation recommended")
        
        # Check for red flag symptoms
        red_flags = []
        if symptoms.get('weight_loss', 0) > 15:
            red_flags.append("severe unexplained weight loss")
        if symptoms.get('fever') and symptoms.get('night_sweats'):
            red_flags.append("B symptoms (fever and night sweats)")
        if symptoms.get('difficulty_swallowing'):
            red_flags.append("dysphagia")
        
        if red_flags:
            warnings.append(f"Red flag symptoms detected: {', '.join(red_flags)}")
        
        # Gender-specific warnings
        if gender == 'Female' and age > 40:
            if symptoms.get('breast_changes'):
                warnings.append("Breast changes in women over 40 require immediate evaluation")
        
        if gender == 'Male' and age > 50:
            if symptoms.get('urinary_changes'):
                warnings.append("Urinary changes in men over 50 may indicate prostate issues")
        
        # Age-specific warnings
        if age and age > 65:
            symptom_count = sum([
                symptoms.get('fatigue', 0) > 5,
                symptoms.get('pain', 0) > 5,
                symptoms.get('weight_loss', 0) > 5,
                symptoms.get('fever', False),
                symptoms.get('night_sweats', False)
            ])
            
            if symptom_count >= 3:
                warnings.append("Multiple symptoms in elderly patient - comprehensive evaluation needed")
        
        return warnings
    
    def _validate_medical_logic(self, patient_data):
        """Validate medical logic and provide clinical insights"""
        warnings = []
        
        age = patient_data.get('age', 0)
        gender = patient_data.get('gender', '')
        smoking = patient_data.get('smoking_history', '')
        symptoms = patient_data.get('symptoms', {})
        family_history = patient_data.get('family_history', [])
        
        # Smoking and respiratory symptoms
        if smoking in ['Current', 'Former'] and (symptoms.get('persistent_cough') or symptoms.get('chest_pain')):
            warnings.append("Smoking history with respiratory symptoms increases lung cancer risk")
        
        # Family history correlations
        if family_history and 'None' not in family_history:
            strong_family_history = len([h for h in family_history if h != 'None']) >= 2
            if strong_family_history:
                warnings.append("Strong family history of cancer - genetic counseling may be beneficial")
        
        # Age and symptom correlation
        if age > 60:
            concerning_symptoms = sum([
                symptoms.get('fatigue', 0) > 6,
                symptoms.get('weight_loss', 0) > 8,
                symptoms.get('pain', 0) > 6
            ])
            
            if concerning_symptoms >= 2:
                warnings.append("Age and symptom profile suggest need for comprehensive cancer screening")
        
        # Weight loss patterns
        weight_loss = symptoms.get('weight_loss', 0)
        if weight_loss > 5:
            if symptoms.get('loss_of_appetite'):
                warnings.append("Weight loss with appetite loss may indicate underlying malignancy")
        
        # Constitutional symptoms (B symptoms)
        constitutional_symptoms = sum([
            symptoms.get('fever', False),
            symptoms.get('night_sweats', False),
            weight_loss > 10
        ])
        
        if constitutional_symptoms >= 2:
            warnings.append("Constitutional symptoms present - may indicate hematologic malignancy")
        
        return warnings
    
    def validate_treatment_data(self, treatment_data):
        """Validate treatment planning data"""
        errors = []
        warnings = []
        
        try:
            # Required treatment fields
            required_fields = ['cancer_type', 'stage', 'patient_performance_status']
            
            for field in required_fields:
                if field not in treatment_data:
                    errors.append(f"Missing required treatment field: {field}")
            
            # Validate performance status (0-4 scale)
            performance_status = treatment_data.get('patient_performance_status')
            if performance_status is not None:
                if not isinstance(performance_status, int) or not (0 <= performance_status <= 4):
                    errors.append("Performance status must be integer between 0-4")
                elif performance_status >= 3:
                    warnings.append("Poor performance status may limit treatment options")
            
            # Validate stage format
            stage = treatment_data.get('stage', '')
            valid_stages = ['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
            if stage not in valid_stages:
                errors.append(f"Stage must be one of: {', '.join(valid_stages)}")
            
            # Treatment timing validation
            if 'start_date' in treatment_data:
                start_date = treatment_data['start_date']
                try:
                    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
                    if start_datetime < datetime.now():
                        warnings.append("Treatment start date is in the past")
                except ValueError:
                    errors.append("Start date must be in YYYY-MM-DD format")
            
        except Exception as e:
            errors.append(f"Treatment validation error: {str(e)}")
        
        return {
            'is_valid': len(errors) == 0,
            'error': '; '.join(errors) if errors else None,
            'warnings': warnings
        }
    
    def validate_progression_data(self, progression_data):
        """Validate disease progression data"""
        errors = []
        warnings = []
        
        try:
            # Validate progression timeline
            if 'timeline' in progression_data:
                timeline = progression_data['timeline']
                if not isinstance(timeline, list):
                    errors.append("Timeline must be a list of milestones")
                else:
                    for i, milestone in enumerate(timeline):
                        if not isinstance(milestone, dict):
                            errors.append(f"Timeline milestone {i+1} must be a dictionary")
                        elif 'date' not in milestone or 'description' not in milestone:
                            errors.append(f"Timeline milestone {i+1} missing required fields")
            
            # Validate progression months
            progression_months = progression_data.get('predicted_progression_months')
            if progression_months is not None:
                if not isinstance(progression_months, (int, float)) or progression_months <= 0:
                    errors.append("Progression months must be a positive number")
                elif progression_months < 3:
                    warnings.append("Very rapid progression predicted - urgent intervention may be needed")
                elif progression_months > 120:
                    warnings.append("Very slow progression predicted - long-term monitoring required")
            
            # Validate confidence intervals
            if 'confidence_interval' in progression_data:
                ci = progression_data['confidence_interval']
                if not isinstance(ci, (list, tuple)) or len(ci) != 2:
                    errors.append("Confidence interval must be a list/tuple with 2 values")
                elif ci[0] >= ci[1]:
                    errors.append("Confidence interval lower bound must be less than upper bound")
        
        except Exception as e:
            errors.append(f"Progression validation error: {str(e)}")
        
        return {
            'is_valid': len(errors) == 0,
            'error': '; '.join(errors) if errors else None,
            'warnings': warnings
        }
    
    def sanitize_input(self, input_data):
        """Sanitize user input to prevent security issues"""
        if isinstance(input_data, str):
            # Remove potentially harmful characters
            sanitized = re.sub(r'[<>"\']', '', input_data)
            # Limit length
            sanitized = sanitized[:1000]
            return sanitized.strip()
        
        elif isinstance(input_data, dict):
            return {key: self.sanitize_input(value) for key, value in input_data.items()}
        
        elif isinstance(input_data, list):
            return [self.sanitize_input(item) for item in input_data]
        
        else:
            return input_data
    
    def check_data_completeness(self, patient_data):
        """Check completeness of patient data for optimal predictions"""
        completeness_score = 0
        missing_optional = []
        
        # Core demographic data (40% weight)
        core_fields = ['age', 'gender', 'weight', 'height']
        core_present = sum(1 for field in core_fields if field in patient_data and patient_data[field] is not None)
        completeness_score += (core_present / len(core_fields)) * 40
        
        # Risk factors (30% weight)
        risk_fields = ['smoking_history', 'alcohol_consumption', 'family_history']
        risk_present = sum(1 for field in risk_fields if field in patient_data and patient_data[field])
        completeness_score += (risk_present / len(risk_fields)) * 30
        
        # Symptoms (30% weight)
        symptoms = patient_data.get('symptoms', {})
        symptom_count = len([k for k, v in symptoms.items() if v])
        max_symptoms = 15  # Approximate number of tracked symptoms
        completeness_score += min(symptom_count / max_symptoms, 1.0) * 30
        
        # Identify missing optional but useful data
        if not patient_data.get('family_history'):
            missing_optional.append("Family cancer history")
        
        if not symptoms.get('weight_loss'):
            missing_optional.append("Recent weight changes")
        
        return {
            'completeness_score': round(completeness_score, 1),
            'completeness_level': self._get_completeness_level(completeness_score),
            'missing_optional_data': missing_optional
        }
    
    def _get_completeness_level(self, score):
        """Determine completeness level based on score"""
        if score >= 90:
            return "Excellent - Optimal for accurate predictions"
        elif score >= 75:
            return "Good - Sufficient for reliable predictions"
        elif score >= 60:
            return "Moderate - Basic predictions possible"
        else:
            return "Limited - Additional data recommended for accuracy"
