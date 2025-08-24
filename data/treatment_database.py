import numpy as np
from datetime import datetime, timedelta

class TreatmentDatabase:
    def __init__(self):
        self.treatment_protocols = self._initialize_treatment_database()
    
    def _initialize_treatment_database(self):
        """Initialize comprehensive treatment database"""
        return {
            'Lung': {
                'Stage I': {
                    'chemotherapy': {
                        'protocol': 'Adjuvant Carboplatin + Paclitaxel',
                        'cycles': [
                            {
                                'cycle_number': 1,
                                'drugs': ['Carboplatin', 'Paclitaxel'],
                                'dosage': 'Carboplatin AUC 6, Paclitaxel 200mg/m²',
                                'duration_days': 1,
                                'interval_weeks': 3,
                                'expected_severity': 'Moderate'
                            },
                            {
                                'cycle_number': 2,
                                'drugs': ['Carboplatin', 'Paclitaxel'],
                                'dosage': 'Carboplatin AUC 6, Paclitaxel 200mg/m²',
                                'duration_days': 1,
                                'interval_weeks': 3,
                                'expected_severity': 'Moderate'
                            },
                            {
                                'cycle_number': 3,
                                'drugs': ['Carboplatin', 'Paclitaxel'],
                                'dosage': 'Carboplatin AUC 6, Paclitaxel 200mg/m²',
                                'duration_days': 1,
                                'interval_weeks': 3,
                                'expected_severity': 'Moderate'
                            },
                            {
                                'cycle_number': 4,
                                'drugs': ['Carboplatin', 'Paclitaxel'],
                                'dosage': 'Carboplatin AUC 6, Paclitaxel 200mg/m²',
                                'duration_days': 1,
                                'interval_weeks': 3,
                                'expected_severity': 'Moderate'
                            }
                        ],
                        'total_weeks': 12
                    },
                    'success_rate': 0.85,
                    'side_effects': [
                        'Neutropenia',
                        'Fatigue',
                        'Nausea/Vomiting',
                        'Peripheral neuropathy',
                        'Hair loss',
                        'Anemia'
                    ],
                    'monitoring': [
                        'Complete blood count weekly',
                        'Liver function tests every 3 weeks',
                        'CT scan every 9 weeks',
                        'Neurological assessment every 3 weeks'
                    ]
                },
                'Stage II': {
                    'chemotherapy': {
                        'protocol': 'Cisplatin + Etoposide',
                        'cycles': [
                            {
                                'cycle_number': 1,
                                'drugs': ['Cisplatin', 'Etoposide'],
                                'dosage': 'Cisplatin 75mg/m², Etoposide 100mg/m²',
                                'duration_days': 3,
                                'interval_weeks': 3,
                                'expected_severity': 'Moderate-Severe'
                            },
                            {
                                'cycle_number': 2,
                                'drugs': ['Cisplatin', 'Etoposide'],
                                'dosage': 'Cisplatin 75mg/m², Etoposide 100mg/m²',
                                'duration_days': 3,
                                'interval_weeks': 3,
                                'expected_severity': 'Moderate-Severe'
                            },
                            {
                                'cycle_number': 3,
                                'drugs': ['Cisplatin', 'Etoposide'],
                                'dosage': 'Cisplatin 75mg/m², Etoposide 100mg/m²',
                                'duration_days': 3,
                                'interval_weeks': 3,
                                'expected_severity': 'Moderate-Severe'
                            },
                            {
                                'cycle_number': 4,
                                'drugs': ['Cisplatin', 'Etoposide'],
                                'dosage': 'Cisplatin 75mg/m², Etoposide 100mg/m²',
                                'duration_days': 3,
                                'interval_weeks': 3,
                                'expected_severity': 'Severe'
                            }
                        ],
                        'total_weeks': 12
                    },
                    'success_rate': 0.70,
                    'side_effects': [
                        'Severe neutropenia',
                        'Nephrotoxicity',
                        'Ototoxicity',
                        'Severe nausea/vomiting',
                        'Hair loss',
                        'Peripheral neuropathy'
                    ],
                    'monitoring': [
                        'Complete blood count twice weekly',
                        'Kidney function tests weekly',
                        'Audiometry every 6 weeks',
                        'CT scan every 6 weeks'
                    ]
                },
                'Stage III': {
                    'chemotherapy': {
                        'protocol': 'Concurrent chemoradiation + Durvalumab',
                        'cycles': [
                            {
                                'cycle_number': 1,
                                'drugs': ['Carboplatin', 'Paclitaxel', 'Durvalumab'],
                                'dosage': 'Carboplatin AUC 6, Paclitaxel 45mg/m² weekly, Durvalumab 10mg/kg',
                                'duration_days': 5,
                                'interval_weeks': 2,
                                'expected_severity': 'Severe'
                            }
                        ] * 6,
                        'total_weeks': 12
                    },
                    'success_rate': 0.50,
                    'side_effects': [
                        'Severe neutropenia',
                        'Esophagitis',
                        'Pneumonitis',
                        'Severe fatigue',
                        'Immune-related adverse events',
                        'Skin reactions'
                    ],
                    'monitoring': [
                        'Complete blood count twice weekly',
                        'Pulmonary function tests monthly',
                        'Thyroid function every 6 weeks',
                        'CT scan every 6 weeks'
                    ]
                },
                'Stage IV': {
                    'chemotherapy': {
                        'protocol': 'Pembrolizumab + Chemotherapy',
                        'cycles': [
                            {
                                'cycle_number': 1,
                                'drugs': ['Pembrolizumab', 'Carboplatin', 'Pemetrexed'],
                                'dosage': 'Pembrolizumab 200mg, Carboplatin AUC 5, Pemetrexed 500mg/m²',
                                'duration_days': 1,
                                'interval_weeks': 3,
                                'expected_severity': 'Severe'
                            }
                        ] * 4,
                        'total_weeks': 12
                    },
                    'success_rate': 0.30,
                    'side_effects': [
                        'Severe immune-related toxicities',
                        'Severe neutropenia',
                        'Pneumonitis',
                        'Colitis',
                        'Hepatitis',
                        'Endocrine dysfunction'
                    ],
                    'monitoring': [
                        'Complete blood count weekly',
                        'Comprehensive metabolic panel weekly',
                        'Thyroid function every 6 weeks',
                        'CT scan every 9 weeks'
                    ]
                }
            },
            'Breast': {
                'Stage I': {
                    'chemotherapy': {
                        'protocol': 'AC-T (Adriamycin/Cyclophosphamide followed by Taxol)',
                        'cycles': [
                            {
                                'cycle_number': 1,
                                'drugs': ['Adriamycin', 'Cyclophosphamide'],
                                'dosage': 'Adriamycin 60mg/m², Cyclophosphamide 600mg/m²',
                                'duration_days': 1,
                                'interval_weeks': 2,
                                'expected_severity': 'Moderate'
                            }
                        ] * 4 + [
                            {
                                'cycle_number': 5,
                                'drugs': ['Paclitaxel'],
                                'dosage': 'Paclitaxel 175mg/m²',
                                'duration_days': 1,
                                'interval_weeks': 2,
                                'expected_severity': 'Moderate'
                            }
                        ] * 4,
                        'total_weeks': 16
                    },
                    'success_rate': 0.90,
                    'side_effects': [
                        'Hair loss',
                        'Nausea/vomiting',
                        'Neutropenia',
                        'Fatigue',
                        'Peripheral neuropathy',
                        'Cardiac toxicity'
                    ],
                    'monitoring': [
                        'Complete blood count every 2 weeks',
                        'Echocardiogram every 12 weeks',
                        'Liver function tests every 6 weeks'
                    ]
                }
                # Additional breast cancer stages would be defined similarly
            },
            'Colorectal': {
                'Stage II': {
                    'chemotherapy': {
                        'protocol': 'FOLFOX (5-FU/Leucovorin/Oxaliplatin)',
                        'cycles': [
                            {
                                'cycle_number': 1,
                                'drugs': ['5-Fluorouracil', 'Leucovorin', 'Oxaliplatin'],
                                'dosage': '5-FU 400mg/m² bolus + 2400mg/m² CI, Leucovorin 200mg/m², Oxaliplatin 85mg/m²',
                                'duration_days': 2,
                                'interval_weeks': 2,
                                'expected_severity': 'Moderate'
                            }
                        ] * 12,
                        'total_weeks': 24
                    },
                    'success_rate': 0.75,
                    'side_effects': [
                        'Peripheral neuropathy',
                        'Neutropenia',
                        'Diarrhea',
                        'Nausea/vomiting',
                        'Hand-foot syndrome',
                        'Mucositis'
                    ],
                    'monitoring': [
                        'Complete blood count every 2 weeks',
                        'Neurological assessment every 4 weeks',
                        'CT scan every 8 weeks'
                    ]
                }
                # Additional colorectal stages would be defined
            }
            # Additional cancer types (Prostate, Stomach, etc.) would be defined similarly
        }
    
    def get_treatment_plan(self, cancer_type, stage):
        """Get treatment plan for specific cancer type and stage"""
        if cancer_type in self.treatment_protocols:
            if stage in self.treatment_protocols[cancer_type]:
                return self.treatment_protocols[cancer_type][stage]
            else:
                # Return closest stage if exact match not found
                available_stages = list(self.treatment_protocols[cancer_type].keys())
                if available_stages:
                    return self.treatment_protocols[cancer_type][available_stages[0]]
        
        # Default treatment plan for unknown cancer types
        return self._get_default_treatment_plan()
    
    def _get_default_treatment_plan(self):
        """Default treatment plan for unknown cancer types"""
        return {
            'chemotherapy': {
                'protocol': 'General supportive care',
                'cycles': [
                    {
                        'cycle_number': 1,
                        'drugs': ['Supportive medications'],
                        'dosage': 'As per standard guidelines',
                        'duration_days': 1,
                        'interval_weeks': 4,
                        'expected_severity': 'Mild'
                    }
                ],
                'total_weeks': 4
            },
            'success_rate': 0.60,
            'side_effects': [
                'Fatigue',
                'Nausea',
                'General discomfort'
            ],
            'monitoring': [
                'Regular clinical assessment',
                'Basic laboratory tests',
                'Imaging as needed'
            ]
        }
    
    def calculate_treatment_schedule(self, treatment_plan, start_date=None):
        """Calculate detailed treatment schedule with dates"""
        if start_date is None:
            start_date = datetime.now()
        
        schedule = []
        current_date = start_date
        
        for cycle in treatment_plan['chemotherapy']['cycles']:
            cycle_info = {
                'cycle_number': cycle['cycle_number'],
                'start_date': current_date.strftime('%Y-%m-%d'),
                'end_date': (current_date + timedelta(days=cycle['duration_days'] - 1)).strftime('%Y-%m-%d'),
                'drugs': cycle['drugs'],
                'dosage': cycle['dosage'],
                'duration_days': cycle['duration_days'],
                'expected_severity': cycle['expected_severity'],
                'pre_medications': self._get_pre_medications(cycle['drugs']),
                'post_medications': self._get_post_medications(cycle['expected_severity'])
            }
            
            schedule.append(cycle_info)
            current_date += timedelta(weeks=cycle['interval_weeks'])
        
        return schedule
    
    def _get_pre_medications(self, drugs):
        """Get pre-medications based on chemotherapy drugs"""
        pre_meds = []
        
        if 'Cisplatin' in drugs:
            pre_meds.extend(['Normal saline hydration', 'Mannitol', 'Magnesium sulfate'])
        
        if any(drug in drugs for drug in ['Adriamycin', 'Paclitaxel', 'Carboplatin']):
            pre_meds.extend(['Dexamethasone', 'Diphenhydramine', 'H2 blocker'])
        
        if 'Paclitaxel' in drugs:
            pre_meds.append('Cimetidine')
        
        return pre_meds
    
    def _get_post_medications(self, severity):
        """Get post-chemotherapy supportive medications"""
        post_meds = []
        
        if severity in ['Moderate', 'Moderate-Severe', 'Severe']:
            post_meds.extend([
                'Anti-nausea medications (Ondansetron)',
                'Proton pump inhibitor',
                'Mouth care protocol'
            ])
        
        if severity in ['Moderate-Severe', 'Severe']:
            post_meds.extend([
                'G-CSF support if needed',
                'Anti-diarrheal medications',
                'Electrolyte monitoring'
            ])
        
        if severity == 'Severe':
            post_meds.extend([
                'Infection prophylaxis',
                'Nutritional support',
                'Close monitoring protocols'
            ])
        
        return post_meds
    
    def get_alternative_treatments(self, cancer_type, stage):
        """Get alternative treatment options"""
        alternatives = {
            'immunotherapy': self._get_immunotherapy_options(cancer_type),
            'targeted_therapy': self._get_targeted_therapy_options(cancer_type),
            'radiation_therapy': self._get_radiation_options(cancer_type, stage),
            'surgical_options': self._get_surgical_options(cancer_type, stage)
        }
        
        return alternatives
    
    def _get_immunotherapy_options(self, cancer_type):
        """Get immunotherapy options by cancer type"""
        immunotherapy_options = {
            'Lung': ['Pembrolizumab', 'Nivolumab', 'Atezolizumab', 'Durvalumab'],
            'Breast': ['Pembrolizumab (triple negative)'],
            'Colorectal': ['Pembrolizumab (MSI-H)', 'Nivolumab (MSI-H)'],
            'Prostate': ['Sipuleucel-T'],
            'Stomach': ['Nivolumab', 'Pembrolizumab']
        }
        
        return immunotherapy_options.get(cancer_type, ['Consult oncologist for options'])
    
    def _get_targeted_therapy_options(self, cancer_type):
        """Get targeted therapy options by cancer type"""
        targeted_options = {
            'Lung': ['Erlotinib', 'Gefitinib', 'Crizotinib', 'Alectinib'],
            'Breast': ['Trastuzumab', 'Pertuzumab', 'CDK4/6 inhibitors'],
            'Colorectal': ['Cetuximab', 'Bevacizumab', 'Regorafenib'],
            'Prostate': ['Enzalutamide', 'Abiraterone', 'Radium-223'],
            'Stomach': ['Trastuzumab (HER2+)', 'Ramucirumab']
        }
        
        return targeted_options.get(cancer_type, ['Molecular testing recommended'])
    
    def _get_radiation_options(self, cancer_type, stage):
        """Get radiation therapy options"""
        if stage in ['Stage I', 'Stage II']:
            return ['External beam radiation therapy', 'Stereotactic body radiation therapy']
        elif stage == 'Stage III':
            return ['Concurrent chemoradiation', 'Sequential chemoradiation']
        else:
            return ['Palliative radiation therapy', 'Stereotactic radiosurgery for metastases']
    
    def _get_surgical_options(self, cancer_type, stage):
        """Get surgical options"""
        if stage in ['Stage 0', 'Stage I', 'Stage II']:
            return ['Curative resection', 'Minimally invasive surgery', 'Robotic surgery']
        elif stage == 'Stage III':
            return ['Neoadjuvant therapy followed by surgery', 'Debulking surgery']
        else:
            return ['Palliative surgery', 'Symptom-relieving procedures']
