import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import colorsys

class VisualizationHelper:
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'danger': '#d62728',
            'info': '#17becf',
            'light': '#7f7f7f',
            'dark': '#393b79'
        }
        
        # Medical color scheme
        self.medical_colors = {
            'stage_0': '#2ecc71',  # Green - early/good
            'stage_1': '#f39c12',  # Orange - moderate
            'stage_2': '#e67e22',  # Dark orange - concerning
            'stage_3': '#e74c3c',  # Red - serious
            'stage_4': '#8e44ad',  # Purple - critical
            'risk_low': '#27ae60',
            'risk_moderate': '#f39c12',
            'risk_high': '#e74c3c',
            'treatment': '#3498db',
            'progression': '#9b59b6',
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'danger': '#d62728',
            'info': '#17becf',
            'light': '#7f7f7f',
            'dark': '#393b79'
        }

    def create_risk_factors_chart(self, risk_factors):
        """Create a horizontal bar chart for risk factors"""
        if not risk_factors:
            return self._create_empty_chart("No risk factors data available")
        
        factors = list(risk_factors.keys())
        values = list(risk_factors.values())
        
        # Determine colors based on risk levels
        colors = []
        for value in values:
            if value >= 0.8:
                colors.append(self.medical_colors['risk_high'])
            elif value >= 0.5:
                colors.append(self.medical_colors['risk_moderate'])
            else:
                colors.append(self.medical_colors['risk_low'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=factors,
            x=values,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition='inside',
            textfont=dict(color='white', size=12),
            hovertemplate='<b>%{y}</b><br>Risk Level: %{x:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Risk Factor Assessment',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Risk Level',
            yaxis_title='Risk Factors',
            xaxis=dict(
                tickformat='.0%',
                range=[0, 1],
                gridcolor='lightgray'
            ),
            yaxis=dict(
                autorange="reversed"
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_staging_chart(self, stage_prediction):
        """Create a staging visualization with survival rates"""
        stages = ['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
        probabilities = []
        colors = []
        
        # Get probabilities for all stages
        all_probs = stage_prediction.get('all_probabilities', {})
        predicted_stage = stage_prediction.get('stage', 'Stage I')
        
        for stage in stages:
            prob = all_probs.get(stage, 0)
            probabilities.append(prob)
            
            # Color coding based on stage severity
            if stage == predicted_stage:
                colors.append(self.medical_colors.get('warning', '#FF9800'))
            elif 'Stage 0' in stage or 'Stage I' in stage:
                colors.append(self.medical_colors.get('stage_0', '#00BFFF'))
            elif 'Stage II' in stage:
                colors.append(self.medical_colors['stage_1'])
            elif 'Stage III' in stage:
                colors.append(self.medical_colors['stage_3'])
            else:
                colors.append(self.medical_colors['stage_4'])
        
        # Create subplot with two charts
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Stage Probability', 'Survival Rates by Stage'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Stage probability chart
        fig.add_trace(
            go.Bar(
                x=stages,
                y=probabilities,
                marker_color=colors,
                text=[f"{p:.1%}" for p in probabilities],
                textposition='outside',
                name='Probability',
                hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Survival rates (typical 5-year survival rates)
        survival_rates = [0.99, 0.95, 0.85, 0.65, 0.25]  # Typical survival rates by stage
        fig.add_trace(
            go.Bar(
                x=stages,
                y=survival_rates,
                marker_color=[self.color_palette['success'] if sr > 0.8 else
                            self.color_palette['warning'] if sr > 0.5 else
                            self.color_palette['danger'] for sr in survival_rates],
                text=[f"{sr:.0%}" for sr in survival_rates],
                textposition='outside',
                name='5-Year Survival',
                hovertemplate='<b>%{x}</b><br>5-Year Survival: %{y:.0%}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title={
                'text': f'Cancer Stage Assessment - Predicted: {predicted_stage}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=False,
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Probability", tickformat='.0%', row=1, col=1)
        fig.update_yaxes(title_text="Survival Rate", tickformat='.0%', row=1, col=2)
        
        return fig
    
    def create_progression_timeline(self, progression_data):
        if not progression_data or 'timeline' not in progression_data:
            return self._create_empty_chart("No progression timeline data available")
        
        timeline = progression_data['timeline']
        if not timeline:
            return self._create_empty_chart("No timeline milestones available")
        
        # Prepare data
        dates = []
        descriptions = []
        likelihoods = []
        
        for milestone in timeline:
            try:
                date_str = milestone.get('date', '')
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date_obj)
                descriptions.append(milestone.get('description', 'Unknown milestone'))
                likelihoods.append(milestone.get('likelihood', 0.5))
            except (ValueError, KeyError):
                continue
        
        if not dates:
            return self._create_empty_chart("No valid milestone dates available")
        
        # Create figure
        fig = go.Figure()

        # Add scatter trace
        fig.add_trace(go.Scatter(
            x=dates,
            y=[1] * len(dates),
            mode='markers+lines+text',
            marker=dict(
                size=[15 + l*20 for l in likelihoods],
                color=[self.medical_colors['progression'] if l > 0.7 else 
                    self.medical_colors['warning'] if l > 0.4 else 
                    self.medical_colors['info'] for l in likelihoods],
                line=dict(width=2, color='white')
            ),
            line=dict(color=self.medical_colors['progression'], width=3),
            text=descriptions,
            textposition='top center',
            textfont=dict(size=10),
            hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Likelihood: %{customdata:.0%}<extra></extra>',
            customdata=[f"{l:.0%}" for l in likelihoods],
            name='Progression Milestones'
        ))
        
        # Handle confidence interval and prediction line
        if 'confidence_interval' in progression_data and 'predicted_progression_months' in progression_data:
            try:
                ci_lower, ci_upper = progression_data['confidence_interval']
                predicted_months = progression_data.get('predicted_progression_months', 12)
                
                base_date = datetime.now()
                
                # Calculate date objects
                ci_lower_date = base_date + timedelta(days=int(ci_lower) * 30)
                ci_upper_date = base_date + timedelta(days=int(ci_upper) * 30)
                predicted_date = base_date + timedelta(days=int(predicted_months) * 30)
                
                # Add confidence interval as a shaded region
                fig.add_shape(
                    type="rect",
                    x0=ci_lower_date,
                    x1=ci_upper_date,
                    y0=0.5,
                    y1=1.5,
                    fillcolor=self.medical_colors['progression'],
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
                
                # Add annotation for confidence interval
                fig.add_annotation(
                    x=ci_lower_date + (ci_upper_date - ci_lower_date) / 2,
                    y=1.4,
                    text="Confidence Interval",
                    showarrow=False,
                    font=dict(size=10)
                )
                
                # Add prediction line
                fig.add_shape(
                    type="line",
                    x0=predicted_date,
                    x1=predicted_date,
                    y0=0.5,
                    y1=1.5,
                    line=dict(
                        color=self.medical_colors['danger'],
                        width=2,
                        dash="dash"
                    )
                )
                
                # Add annotation for prediction
                fig.add_annotation(
                    x=predicted_date,
                    y=1.4,
                    text=f"Predicted Progression<br>({predicted_months:.1f} months)",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=self.medical_colors['danger'],
                    font=dict(size=10)
                )
                
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not add confidence interval or prediction line: {e}")
        
        fig.update_xaxes(
            type='date',
            title='Timeline'
        )
        
        fig.update_yaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0.5, 1.5]
        )
        
        fig.update_layout(
            title={
                'text': 'Disease Progression Timeline',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_treatment_timeline(self, schedule_data):
        """Create a Gantt-style chart for treatment schedule"""
        if not schedule_data:
            return self._create_empty_chart("No treatment schedule data available")
        
        # Prepare data for Gantt chart
        df_schedule = pd.DataFrame(schedule_data)
        
        # Convert dates
        df_schedule['Date'] = pd.to_datetime(df_schedule['Date'])
        df_schedule['End_Date'] = df_schedule['Date'] + pd.to_timedelta(df_schedule['Duration'].str.extract('(\d+)')[0].astype(int), unit='D')
        
        # Color mapping for severity
        severity_colors = {
            'Mild': self.medical_colors['risk_low'],
            'Moderate': self.medical_colors['risk_moderate'],
            'Moderate-Severe': self.medical_colors['warning'],
            'Severe': self.medical_colors['risk_high']
        }
        
        fig = go.Figure()
        
        for idx, row in df_schedule.iterrows():
            cycle_name = f"Cycle {row['Cycle']}"
            color = severity_colors.get(row['Severity'], self.medical_colors['treatment'])
            
            fig.add_trace(go.Scatter(
                x=[row['Date'], row['End_Date']],
                y=[cycle_name, cycle_name],
                mode='lines+markers',
                line=dict(color=color, width=8),
                marker=dict(size=12, color=color),
                hovertemplate=f'<b>{cycle_name}</b><br>' +
                            f'Drugs: {row["Drugs"]}<br>' +
                            f'Duration: {row["Duration"]}<br>' +
                            f'Severity: {row["Severity"]}<br>' +
                            f'Start: %{{x}}<extra></extra>',
                name=cycle_name,
                showlegend=False
            ))
        
        # Add severity legend
        for severity, color in severity_colors.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=12, color=color),
                name=severity,
                showlegend=True
            ))
        
        fig.update_layout(
            title={
                'text': 'Treatment Schedule Timeline',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Date',
            yaxis_title='Treatment Cycles',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            margin=dict(l=100, r=50, t=80, b=50),
            legend=dict(
                title="Expected Severity",
                x=1.02,
                y=1
            )
        )
        
        return fig
    
    def create_comprehensive_risk_timeline(self, diagnosis, patient_data):
        """Create comprehensive risk assessment over time"""
        # Generate risk progression over time
        months = list(range(0, 25, 3))  # 0 to 24 months, every 3 months
        base_risk = diagnosis['cancer_prediction']['probability']
        stage = diagnosis['stage_prediction']['stage']
        
        # Calculate risk progression based on stage
        stage_multipliers = {
            'Stage 0': 0.1,
            'Stage I': 0.3,
            'Stage II': 0.5,
            'Stage III': 0.8,
            'Stage IV': 1.2
        }
        
        multiplier = stage_multipliers.get(stage, 0.5)
        risk_values = []
        
        for month in months:
            # Risk increases over time without treatment
            progression_factor = 1 + (month * multiplier * 0.05)
            risk = min(base_risk * progression_factor, 1.0)
            risk_values.append(risk)
        
        # Create the risk timeline chart
        fig = go.Figure()
        
        # Add risk progression line
        fig.add_trace(go.Scatter(
            x=months,
            y=risk_values,
            mode='lines+markers',
            name='Disease Risk',
            line=dict(color=self.medical_colors['danger'], width=3),
            marker=dict(size=8),
            hovertemplate='Month %{x}<br>Risk Level: %{y:.1%}<extra></extra>'
        ))
        
        # Add risk zones
        fig.add_hrect(
            y0=0, y1=0.3,
            fillcolor=self.medical_colors['risk_low'],
            opacity=0.2,
            layer="below",
            annotation_text="Low Risk Zone",
            annotation_position="bottom left"
        )
        
        fig.add_hrect(
            y0=0.3, y1=0.7,
            fillcolor=self.medical_colors['risk_moderate'],
            opacity=0.2,
            layer="below",
            annotation_text="Moderate Risk Zone",
            annotation_position="left"
        )
        
        fig.add_hrect(
            y0=0.7, y1=1.0,
            fillcolor=self.medical_colors['risk_high'],
            opacity=0.2,
            layer="below",
            annotation_text="High Risk Zone",
            annotation_position="top left"
        )
        
        # Add current risk point
        fig.add_trace(go.Scatter(
            x=[0],
            y=[base_risk],
            mode='markers',
            name='Current Risk',
            marker=dict(
                size=15,
                color=self.medical_colors['warning'],
                symbol='star'
            ),
            hovertemplate='Current Risk: %{y:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Risk Assessment Timeline (Without Treatment)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Months from Now',
            yaxis_title='Risk Level',
            yaxis=dict(
                tickformat='.0%',
                range=[0, 1]
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            legend=dict(x=1.02, y=1)
        )
        
        return fig
    
    def create_treatment_effectiveness_chart(self, treatment_plan):
        """Create treatment effectiveness visualization"""
        if not treatment_plan or 'plan' not in treatment_plan:
            return self._create_empty_chart("No treatment plan data available")
        
        plan = treatment_plan['plan']
        success_rate = plan.get('success_rate', 0.6)
        
        # Create effectiveness over time chart
        weeks = list(range(0, plan['chemotherapy'].get('total_weeks', 12) + 1, 2))
        effectiveness = []
        
        # Model effectiveness buildup over treatment period
        max_effectiveness = success_rate
        for week in weeks:
            if week == 0:
                effectiveness.append(0)
            else:
                # Sigmoid curve for treatment response
                progress = week / plan['chemotherapy'].get('total_weeks', 12)
                eff = max_effectiveness * (1 / (1 + np.exp(-6 * (progress - 0.5))))
                effectiveness.append(eff)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Treatment Effectiveness Over Time', 'Expected Outcomes'),
            specs=[[{'type': 'scatter'}, {'type': 'pie'}]]
        )
        
        # Effectiveness timeline
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=effectiveness,
                mode='lines+markers',
                name='Treatment Response',
                line=dict(color=self.medical_colors['treatment'], width=3),
                marker=dict(size=8),
                hovertemplate='Week %{x}<br>Effectiveness: %{y:.1%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add target effectiveness line
        fig.add_hline(
            y=success_rate,
            line_dash="dash",
            line_color=self.medical_colors['success'],
            annotation_text=f"Target Success Rate: {success_rate:.0%}",
            row=1, col=1
        )
        
        # Outcome pie chart
        outcomes = ['Complete Response', 'Partial Response', 'Stable Disease', 'Progressive Disease']
        outcome_values = [
            success_rate * 0.4,  # Complete response
            success_rate * 0.6,  # Partial response
            (1 - success_rate) * 0.7,  # Stable disease
            (1 - success_rate) * 0.3   # Progressive disease
        ]
        
        outcome_colors = [
            self.medical_colors['success'],
            self.medical_colors['risk_moderate'],
            self.medical_colors['info'],
            self.medical_colors['risk_high']
        ]
        
        fig.add_trace(
            go.Pie(
                labels=outcomes,
                values=outcome_values,
                marker_colors=outcome_colors,
                hovertemplate='%{label}<br>Probability: %{percent}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title={
                'text': 'Treatment Effectiveness Prediction',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Weeks of Treatment", row=1, col=1)
        fig.update_yaxes(title_text="Treatment Effectiveness", tickformat='.0%', row=1, col=1)
        
        return fig
    
    def create_symptom_severity_chart(self, current_symptoms, predicted_symptoms):
        """Create a comparison chart of current vs predicted symptoms"""
        if not current_symptoms and not predicted_symptoms:
            return self._create_empty_chart("No symptom data available")
        
        # Combine and organize symptom data
        all_symptoms = set()
        if current_symptoms:
            all_symptoms.update(current_symptoms.keys())
        if predicted_symptoms:
            all_symptoms.update([s[0] for s in predicted_symptoms])
        
        symptoms_list = list(all_symptoms)
        current_values = []
        predicted_values = []
        
        for symptom in symptoms_list:
            # Current symptoms
            if current_symptoms and symptom in current_symptoms:
                current_values.append(current_symptoms[symptom])
            else:
                current_values.append(0)
            
            # Predicted symptoms
            pred_value = 0
            if predicted_symptoms:
                for pred_symptom, likelihood, severity in predicted_symptoms:
                    if pred_symptom == symptom:
                        # Convert severity to numeric (assuming 0-10 scale)
                        severity_map = {'Mild': 3, 'Moderate': 6, 'Severe': 9}
                        pred_value = severity_map.get(severity, 5) * likelihood
                        break
            predicted_values.append(pred_value)
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current Symptoms',
            x=symptoms_list,
            y=current_values,
            marker_color=self.medical_colors['info'],
            hovertemplate='%{x}<br>Current Severity: %{y}<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='Predicted Symptoms',
            x=symptoms_list,
            y=predicted_values,
            marker_color=self.medical_colors['warning'],
            hovertemplate='%{x}<br>Predicted Severity: %{y:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Current vs Predicted Symptom Severity',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Symptoms',
            yaxis_title='Severity Level',
            yaxis=dict(range=[0, 10]),
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            legend=dict(x=1.02, y=1)
        )
        
        return fig
    
    def create_nutrition_adherence_chart(self, nutrition_data):
        """Create nutrition adherence and recommendation chart"""
        # This would typically use real nutrition tracking data
        # For now, create a sample adherence chart
        
        categories = ['Proteins', 'Vegetables', 'Fruits', 'Whole Grains', 'Healthy Fats', 'Hydration']
        recommended = [100, 100, 100, 100, 100, 100]  # Target 100%
        actual = [85, 92, 78, 88, 95, 82]  # Sample adherence rates
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Recommended',
            x=categories,
            y=recommended,
            marker_color=self.medical_colors['success'],
            opacity=0.7,
            hovertemplate='%{x}<br>Recommended: %{y}%<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='Actual Intake',
            x=categories,
            y=actual,
            marker_color=self.medical_colors['treatment'],
            hovertemplate='%{x}<br>Actual: %{y}%<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Nutrition Plan Adherence',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Nutrition Categories',
            yaxis_title='Adherence Percentage',
            yaxis=dict(range=[0, 120], tickformat='%'),
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        
        return fig
    
    def _create_empty_chart(self, message):
        """Create an empty chart with a message"""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=message,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16),
            xanchor='center',
            yanchor='middle'
        )
        
        fig.update_layout(
            title={
                'text': 'Data Visualization',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300
        )
        
        return fig
    
    def create_survival_curve(self, stage, cancer_type):
        """Create Kaplan-Meier style survival curve"""
        # Generate survival curve data based on stage
        months = np.arange(0, 61, 1)  # 5 years
        
        # Stage-based survival parameters
        stage_params = {
            'Stage 0': {'median': 60, 'shape': 0.95},
            'Stage I': {'median': 58, 'shape': 0.90},
            'Stage II': {'median': 48, 'shape': 0.75},
            'Stage III': {'median': 36, 'shape': 0.55},
            'Stage IV': {'median': 18, 'shape': 0.25}
        }
        
        params = stage_params.get(stage, stage_params['Stage I'])
        
        # Generate survival probabilities (simplified exponential decay)
        survival_probs = []
        for month in months:
            if month == 0:
                survival_probs.append(1.0)
            else:
                # Exponential survival function
                prob = params['shape'] * np.exp(-month / params['median'])
                survival_probs.append(max(prob, 0.01))  # Minimum 1% survival
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=survival_probs,
            mode='lines',
            name=f'{stage} Survival',
            line=dict(
                color=self.medical_colors['treatment'],
                width=3
            ),
            fill='tonexty',
            fillcolor=f"rgba(52, 152, 219, 0.2)",
            hovertemplate='Month %{x}<br>Survival Probability: %{y:.1%}<extra></extra>'
        ))
        
        # Add median survival line
        median_survival = params['median']
        fig.add_vline(
            x=median_survival,
            line_dash="dash",
            line_color=self.medical_colors['warning'],
            annotation_text=f"Median Survival: {median_survival} months"
        )
        
        fig.update_layout(
            title={
                'text': f'Estimated Survival Curve - {cancer_type} {stage}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Months from Diagnosis',
            yaxis_title='Survival Probability',
            yaxis=dict(
                tickformat='.0%',
                range=[0, 1]
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        
        return fig