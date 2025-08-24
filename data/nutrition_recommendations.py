import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class NutritionRecommendations:
    def __init__(self):
        self.nutrition_database = self._initialize_nutrition_database()
        self.meal_database = self._initialize_meal_database()
    
    def _initialize_nutrition_database(self):
        """Initialize comprehensive nutrition recommendations database"""
        return {
            'Lung': {
                'Stage I': {
                    'recommended_foods': [
                        'Antioxidant-rich fruits (berries, citrus)',
                        'Cruciferous vegetables (broccoli, cauliflower)',
                        'Lean proteins (fish, poultry, legumes)',
                        'Whole grains (quinoa, brown rice)',
                        'Green tea',
                        'Omega-3 rich foods (salmon, walnuts)',
                        'Sweet potatoes',
                        'Leafy greens (spinach, kale)'
                    ],
                    'avoid_foods': [
                        'Processed meats',
                        'Excessive red meat',
                        'Refined sugars',
                        'Alcohol',
                        'High-sodium foods',
                        'Fried foods',
                        'Smoking (tobacco products)'
                    ],
                    'nutritional_goals': [
                        'Maintain healthy weight',
                        'Increase antioxidant intake',
                        'Support immune function',
                        'Reduce inflammation',
                        '2000-2200 calories/day',
                        '1.2-1.5g protein/kg body weight'
                    ],
                    'supplements': [
                        'Vitamin D (1000-2000 IU/day)',
                        'Omega-3 fatty acids',
                        'Multivitamin',
                        'Vitamin C (500mg/day)'
                    ]
                },
                'Stage II': {
                    'recommended_foods': [
                        'High-protein foods for healing',
                        'Soft, easy-to-digest foods',
                        'Nutrient-dense smoothies',
                        'Bone broth',
                        'Cooked vegetables',
                        'Protein shakes',
                        'Avocados',
                        'Nut butters'
                    ],
                    'avoid_foods': [
                        'Raw vegetables (if digestive issues)',
                        'Spicy foods',
                        'Very hot or cold foods',
                        'Alcohol',
                        'High-fiber foods if nauseous',
                        'Strong-smelling foods'
                    ],
                    'nutritional_goals': [
                        'Prevent weight loss',
                        'Maintain protein stores',
                        'Support treatment tolerance',
                        '2200-2500 calories/day',
                        '1.5-2.0g protein/kg body weight',
                        'Stay hydrated (8-10 glasses water/day)'
                    ],
                    'supplements': [
                        'Protein supplements',
                        'B-complex vitamins',
                        'Probiotics',
                        'Ginger for nausea'
                    ]
                },
                'Stage III': {
                    'recommended_foods': [
                        'Liquid nutrition supplements',
                        'Pureed foods',
                        'High-calorie, high-protein foods',
                        'Milkshakes and smoothies',
                        'Cream soups',
                        'Puddings and custards',
                        'Mashed potatoes with butter',
                        'Scrambled eggs with cream'
                    ],
                    'avoid_foods': [
                        'Rough or scratchy textures',
                        'Very hot foods',
                        'Acidic foods',
                        'Spicy foods',
                        'Dry foods',
                        'Hard-to-swallow items'
                    ],
                    'nutritional_goals': [
                        'Prevent severe weight loss',
                        'Maximize calorie intake',
                        'Ease swallowing difficulties',
                        '2500+ calories/day if possible',
                        '2.0g+ protein/kg body weight',
                        'Frequent small meals'
                    ],
                    'supplements': [
                        'Complete liquid nutrition',
                        'Protein powders',
                        'Enzyme supplements',
                        'Appetite stimulants (if prescribed)'
                    ]
                },
                'Stage IV': {
                    'recommended_foods': [
                        'Comfort foods patient enjoys',
                        'High-calorie liquids',
                        'Soft, palatable foods',
                        'Ice cream and sorbet',
                        'Nutritional drinks',
                        'Gelatin and popsicles',
                        'Broths and clear soups',
                        'Favorite childhood foods'
                    ],
                    'avoid_foods': [
                        'Foods that cause discomfort',
                        'Strong odors if nauseous',
                        'Large portions',
                        'Foods requiring much chewing'
                    ],
                    'nutritional_goals': [
                        'Focus on quality of life',
                        'Maintain comfort',
                        'Prevent dehydration',
                        'Small, frequent meals',
                        'Honor food preferences',
                        'Family meal involvement'
                    ],
                    'supplements': [
                        'As tolerated',
                        'Liquid vitamins',
                        'Electrolyte supplements',
                        'Comfort-focused nutrition'
                    ]
                }
            },
            'Breast': {
                'Stage I': {
                    'recommended_foods': [
                        'Soy foods (tofu, edamame)',
                        'Cruciferous vegetables',
                        'Berries and cherries',
                        'Fatty fish (salmon, sardines)',
                        'Flaxseeds and chia seeds',
                        'Green tea',
                        'Pomegranate',
                        'Dark leafy greens'
                    ],
                    'avoid_foods': [
                        'Excessive alcohol',
                        'High-fat dairy',
                        'Processed foods',
                        'Trans fats',
                        'Excessive sugar',
                        'Charred meats'
                    ],
                    'nutritional_goals': [
                        'Maintain healthy weight',
                        'Support hormonal balance',
                        'Increase phytoestrogen intake',
                        '1800-2000 calories/day',
                        'Maintain BMI 18.5-24.9'
                    ]
                }
                # Additional breast cancer stages...
            },
            'Colorectal': {
                'Stage II': {
                    'recommended_foods': [
                        'High-fiber foods (gradually)',
                        'Probiotic-rich foods',
                        'Lean proteins',
                        'Cooked vegetables',
                        'Whole grains (as tolerated)',
                        'Bananas and rice',
                        'Yogurt with live cultures',
                        'Applesauce'
                    ],
                    'avoid_foods': [
                        'High-fat foods',
                        'Spicy foods',
                        'Raw vegetables initially',
                        'Nuts and seeds initially',
                        'Alcohol',
                        'Caffeine excess',
                        'Artificial sweeteners'
                    ],
                    'nutritional_goals': [
                        'Support digestive healing',
                        'Gradual fiber increase',
                        'Maintain hydration',
                        'Support microbiome',
                        '2000-2200 calories/day'
                    ]
                }
            }
            # Additional cancer types would be added here...
        }
    
    def _initialize_meal_database(self):
        """Initialize meal planning database"""
        return {
            'breakfast': {
                'high_protein': [
                    'Scrambled eggs with spinach and cheese',
                    'Greek yogurt with berries and granola',
                    'Protein smoothie with banana and peanut butter',
                    'Oatmeal with protein powder and nuts',
                    'Cottage cheese with fruit',
                    'Avocado toast with eggs'
                ],
                'gentle_digestion': [
                    'Banana and rice porridge',
                    'Plain toast with honey',
                    'Chamomile tea with crackers',
                    'Applesauce with cinnamon',
                    'Rice cereal with milk',
                    'Ginger tea with plain biscuits'
                ],
                'high_calorie': [
                    'Pancakes with butter and syrup',
                    'French toast with cream',
                    'Milkshake with protein powder',
                    'Granola with whole milk and honey',
                    'Bagel with cream cheese',
                    'Breakfast burrito with cheese'
                ]
            },
            'lunch': {
                'high_protein': [
                    'Grilled chicken salad with quinoa',
                    'Tuna sandwich on whole grain bread',
                    'Lentil soup with whole grain roll',
                    'Turkey and avocado wrap',
                    'Bean and vegetable stew',
                    'Salmon with sweet potato'
                ],
                'gentle_digestion': [
                    'Chicken noodle soup',
                    'Plain rice with steamed vegetables',
                    'Mashed potatoes with gravy',
                    'Cream of mushroom soup',
                    'Soft pasta with butter',
                    'Pureed vegetable soup'
                ],
                'high_calorie': [
                    'Cheeseburger with fries',
                    'Pizza with favorite toppings',
                    'Creamy pasta with cheese sauce',
                    'Grilled cheese with tomato soup',
                    'Quesadilla with sour cream',
                    'Fried chicken with mashed potatoes'
                ]
            },
            'dinner': {
                'high_protein': [
                    'Baked salmon with roasted vegetables',
                    'Lean beef stir-fry with brown rice',
                    'Grilled chicken with quinoa pilaf',
                    'Turkey meatballs with marinara',
                    'Tofu curry with vegetables',
                    'Fish tacos with black beans'
                ],
                'gentle_digestion': [
                    'Baked chicken breast with rice',
                    'Vegetable broth with soft noodles',
                    'Steamed fish with mashed carrots',
                    'Plain baked potato with butter',
                    'Soft-cooked vegetables with rice',
                    'Chicken and rice soup'
                ],
                'high_calorie': [
                    'Lasagna with garlic bread',
                    'Steak with loaded baked potato',
                    'Fried fish with coleslaw',
                    'Pork chops with apple stuffing',
                    'Beef stroganoff with noodles',
                    'BBQ ribs with cornbread'
                ]
            },
            'snacks': {
                'high_protein': [
                    'Greek yogurt with nuts',
                    'Protein bars',
                    'Hard-boiled eggs',
                    'String cheese with apple',
                    'Hummus with vegetables',
                    'Trail mix with nuts'
                ],
                'gentle_digestion': [
                    'Banana with honey',
                    'Plain crackers',
                    'Applesauce cups',
                    'Ginger ale with crackers',
                    'Rice cakes',
                    'Popsicles'
                ],
                'high_calorie': [
                    'Ice cream',
                    'Milkshakes',
                    'Chocolate bars',
                    'Cookies and milk',
                    'Cheese and crackers',
                    'Nuts and dried fruit'
                ]
            }
        }
    
    def get_nutrition_plan(self, cancer_type, stage):
        """Get personalized nutrition plan"""
        if cancer_type in self.nutrition_database:
            if stage in self.nutrition_database[cancer_type]:
                return self.nutrition_database[cancer_type][stage]
            else:
                # Return Stage I plan as default
                available_stages = list(self.nutrition_database[cancer_type].keys())
                if available_stages:
                    return self.nutrition_database[cancer_type][available_stages[0]]
        
        # Default nutrition plan
        return self._get_default_nutrition_plan()
    
    def _get_default_nutrition_plan(self):
        """Default nutrition plan for unknown cancer types"""
        return {
            'recommended_foods': [
                'Variety of fruits and vegetables',
                'Lean proteins',
                'Whole grains',
                'Healthy fats',
                'Adequate hydration'
            ],
            'avoid_foods': [
                'Processed foods',
                'Excessive alcohol',
                'High-sugar items',
                'Trans fats'
            ],
            'nutritional_goals': [
                'Maintain balanced diet',
                'Support overall health',
                'Stay hydrated'
            ],
            'supplements': [
                'Multivitamin',
                'As recommended by healthcare provider'
            ]
        }
    
    def generate_weekly_meal_plan(self, nutrition_plan):
        """Generate a weekly meal plan based on nutrition recommendations"""
        # Determine meal plan type based on stage and recommendations
        meal_type = self._determine_meal_type(nutrition_plan)
        
        weekly_plan = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in days:
            daily_meals = {
                'Breakfast': self._select_meal('breakfast', meal_type),
                'Lunch': self._select_meal('lunch', meal_type),
                'Dinner': self._select_meal('dinner', meal_type),
                'Snacks': self._select_meal('snacks', meal_type)
            }
            weekly_plan[day] = daily_meals
        
        return weekly_plan
    
    def _determine_meal_type(self, nutrition_plan):
        """Determine appropriate meal type based on nutrition plan"""
        goals = nutrition_plan.get('nutritional_goals', [])
        
        # Check for high-calorie needs
        if any('calorie' in goal.lower() and ('2500' in goal or 'prevent weight loss' in goal.lower()) 
               for goal in goals):
            return 'high_calorie'
        
        # Check for digestive issues
        recommended_foods = nutrition_plan.get('recommended_foods', [])
        if any('soft' in food.lower() or 'liquid' in food.lower() or 'pureed' in food.lower() 
               for food in recommended_foods):
            return 'gentle_digestion'
        
        # Default to high protein
        return 'high_protein'
    
    def _select_meal(self, meal_category, meal_type):
        """Select appropriate meal from database"""
        if meal_category in self.meal_database and meal_type in self.meal_database[meal_category]:
            meals = self.meal_database[meal_category][meal_type]
            return random.choice(meals)
        else:
            return f"Consult nutritionist for {meal_category} recommendations"
    
    def get_lifestyle_recommendations(self, cancer_type, stage):
        """Get comprehensive lifestyle recommendations"""
        
        # Activity recommendations based on stage
        activity_levels = {
            'Stage 0': {
                'recommended_activities': [
                    'Regular moderate exercise (30 min, 5 days/week)',
                    'Strength training (2-3 times/week)',
                    'Yoga or stretching',
                    'Walking or hiking',
                    'Swimming',
                    'Gardening'
                ],
                'restrictions': [
                    'Avoid overexertion',
                    'Listen to your body',
                    'Stay hydrated during exercise'
                ]
            },
            'Stage I': {
                'recommended_activities': [
                    'Light to moderate exercise as tolerated',
                    'Walking (start with 10-15 minutes)',
                    'Gentle yoga',
                    'Light resistance exercises',
                    'Breathing exercises',
                    'Meditation'
                ],
                'restrictions': [
                    'Avoid contact sports',
                    'No heavy lifting during treatment',
                    'Rest when fatigued',
                    'Avoid crowded places if immunocompromised'
                ]
            },
            'Stage II': {
                'recommended_activities': [
                    'Gentle walking as tolerated',
                    'Chair exercises',
                    'Deep breathing exercises',
                    'Light stretching',
                    'Meditation and mindfulness',
                    'Creative activities (art, music)'
                ],
                'restrictions': [
                    'Significant activity limitations',
                    'Avoid strenuous exercise',
                    'No lifting >10 pounds',
                    'Frequent rest periods needed',
                    'Avoid exposure to infections'
                ]
            },
            'Stage III': {
                'recommended_activities': [
                    'Very gentle movement as tolerated',
                    'Passive stretching',
                    'Breathing exercises',
                    'Meditation',
                    'Light mental activities',
                    'Social interaction as desired'
                ],
                'restrictions': [
                    'Severe activity limitations',
                    'Bed rest may be required',
                    'Assistance needed for daily activities',
                    'Energy conservation is priority',
                    'Avoid all strenuous activities'
                ]
            },
            'Stage IV': {
                'recommended_activities': [
                    'Activities that bring comfort and joy',
                    'Gentle movements as able',
                    'Relaxation techniques',
                    'Spending time with loved ones',
                    'Enjoying favorite pastimes',
                    'Spiritual or meaningful activities'
                ],
                'restrictions': [
                    'Focus on comfort and quality of life',
                    'Activities limited by symptoms',
                    'Assistance needed for most activities',
                    'Energy conservation critical'
                ]
            }
        }
        
        activities = activity_levels.get(stage, activity_levels['Stage I'])
        
        # Sleep recommendations
        sleep_recs = [
            'Maintain regular sleep schedule',
            '7-9 hours of sleep nightly',
            'Create comfortable sleep environment',
            'Limit screen time before bed',
            'Use relaxation techniques',
            'Manage pain and discomfort for better sleep',
            'Short naps (20-30 min) if needed'
        ]
        
        # Stress management
        stress_management = [
            'Practice mindfulness meditation',
            'Deep breathing exercises',
            'Join support groups',
            'Counseling or therapy',
            'Maintain social connections',
            'Express feelings through journaling',
            'Engage in enjoyable activities',
            'Consider complementary therapies (massage, acupuncture)'
        ]
        
        # Adjust recommendations based on stage severity
        if stage in ['Stage III', 'Stage IV']:
            sleep_recs.extend([
                'Frequent rest periods during day',
                'Comfortable positioning for sleep',
                'Medication for sleep if prescribed'
            ])
            
            stress_management.extend([
                'Focus on acceptance and adaptation',
                'Legacy projects or meaningful activities',
                'Spiritual care if desired',
                'Family counseling support'
            ])
        
        return {
            'recommended_activities': activities['recommended_activities'],
            'restrictions': activities['restrictions'],
            'sleep_recommendations': sleep_recs,
            'stress_management': stress_management
        }
    
    def get_hydration_guidelines(self, stage, treatment_status=False):
        """Get hydration guidelines based on stage and treatment"""
        base_fluid = 8  # glasses per day
        
        if stage in ['Stage III', 'Stage IV'] or treatment_status:
            recommendations = [
                f'Aim for {base_fluid + 2}-{base_fluid + 4} glasses of fluid daily',
                'Include water, clear broths, herbal teas',
                'Avoid caffeine and alcohol',
                'Monitor urine color (pale yellow is good)',
                'Sip small amounts frequently',
                'Use electrolyte supplements if recommended'
            ]
        else:
            recommendations = [
                f'Aim for {base_fluid} glasses of water daily',
                'Increase with exercise or hot weather',
                'Include variety: water, herbal teas, broths',
                'Limit caffeinated and alcoholic beverages',
                'Monitor hydration status'
            ]
        
        return recommendations
    
    def get_supplement_guidance(self, cancer_type, stage, current_treatment=None):
        """Get supplement recommendations with medical considerations"""
        
        general_guidance = [
            'Always consult healthcare team before starting supplements',
            'Some supplements may interfere with treatments',
            'Choose reputable brands with third-party testing',
            'Start with one supplement at a time',
            'Monitor for any adverse reactions'
        ]
        
        # Get specific supplement recommendations
        nutrition_plan = self.get_nutrition_plan(cancer_type, stage)
        supplements = nutrition_plan.get('supplements', [])
        
        # Add treatment-specific considerations
        treatment_considerations = []
        if current_treatment:
            treatment_considerations = [
                'Some supplements may reduce chemotherapy effectiveness',
                'Antioxidant supplements timing may need adjustment',
                'Iron supplements may be contraindicated during some treatments',
                'Probiotics may be beneficial during antibiotic therapy'
            ]
        
        return {
            'recommended_supplements': supplements,
            'general_guidance': general_guidance,
            'treatment_considerations': treatment_considerations
        }
