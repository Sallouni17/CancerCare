# Medical AI Cancer Diagnosis System

## Overview

This is a comprehensive medical AI system for cancer diagnosis, staging, progression monitoring, and treatment planning. The system provides an integrated platform combining machine learning models for cancer prediction, staging assessment, and progression forecasting with treatment protocols and nutritional recommendations. Built using Streamlit for the web interface, it offers healthcare professionals a complete toolset for cancer patient management.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Single-page application with multi-module navigation
- **Interactive Dashboard**: Wide layout with sidebar navigation for different modules (Patient Input, Diagnosis Results, Treatment Planning, Lifestyle Management, Reports)
- **Visualization Layer**: Plotly-based charts and graphs for medical data visualization
- **Caching Strategy**: Streamlit resource caching for ML models and data to optimize performance

### Backend Architecture
- **Machine Learning Pipeline**: Three core ML models using scikit-learn:
  - Cancer Predictor: Random Forest and Gradient Boosting for initial cancer detection
  - Staging Model: Random Forest for TNM staging classification
  - Progression Model: Gradient Boosting for disease progression forecasting
- **Data Processing**: StandardScaler and LabelEncoder for feature preprocessing
- **Synthetic Data Generation**: Medical literature-based synthetic datasets for model training
- **Modular Design**: Separation of concerns with distinct modules for models, data, and utilities

### Data Storage Solutions
- **In-Memory Databases**: Python dictionaries and pandas DataFrames for treatment protocols and nutrition recommendations
- **Model Persistence**: Joblib for saving/loading trained ML models
- **Structured Medical Data**: Hierarchical data organization by cancer type and stage
- **No External Database**: Self-contained system with embedded medical knowledge base

### Validation and Quality Assurance
- **Data Validation Layer**: Comprehensive input validation with medical range checking
- **Error Handling**: Structured error and warning system for patient data validation
- **Medical Guidelines Compliance**: Age-specific protocols and treatment recommendations
- **Input Sanitization**: Regex patterns and range validation for all user inputs

### Medical Knowledge Integration
- **Treatment Database**: Evidence-based chemotherapy protocols with cycle management
- **Nutrition Recommendations**: Stage-specific dietary guidelines and supplement recommendations
- **Risk Assessment**: Multi-factor risk scoring based on demographics, lifestyle, and symptoms
- **Progression Modeling**: Time-series forecasting for disease progression and symptom evolution

## External Dependencies

### Core ML and Data Science Stack
- **scikit-learn**: Machine learning algorithms (Random Forest, Gradient Boosting, preprocessing)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations

### Visualization and UI
- **Streamlit**: Web application framework and UI components
- **Plotly**: Interactive charting library (plotly.express and plotly.graph_objects)

### Utility Libraries
- **datetime**: Time-based calculations for treatment scheduling and progression modeling
- **warnings**: Python warning system management
- **re**: Regular expressions for data validation
- **random**: Random number generation for synthetic data
- **colorsys**: Color space conversions for visualization

### Model Persistence
- **joblib**: Model serialization and deserialization for ML models

### Python Standard Library
- **Built-in modules**: Extensive use of Python standard library for core functionality without external API dependencies