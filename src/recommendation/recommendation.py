"""
RecommendationEngine: Core recommendation engine for real-time property recommendations

This module provides contextualized real estate recommendations based on user roles,
property data, and model predictions. It serves as the bridge between models and
user-facing recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class RecommendationEngine:
    """
    A context-aware recommendation engine for real estate that adapts recommendations
    based on user type (owner, investor, agent, tenant).
    """
    
    def __init__(self):
        """Initialize the recommendation engine."""
        self.user_contexts = ['owner', 'investor', 'agent', 'tenant']
        self.recommendation_history = {}
        self.decision_thresholds = {
            'high_yield': 7.0,  # Rental yield above 7% is considered high
            'undervalue_threshold': 10.0,  # Property undervalued by >10% 
            'rent_increase_limit': 15.0,  # Maximum suggested rent increase
            'rent_decrease_limit': 10.0,  # Maximum suggested rent decrease
            'price_to_rent_good': 15.0,  # P/R ratio below 15 is good for investment
            'location_premium_threshold': 0.1  # Location is premium if >10% above avg
        }
    
    def get_recommendations(self, property_data, model_results, user_context='owner', user_id=None):
        """
        Generate context-aware recommendations based on user type.
        
        Parameters:
            property_data (dict or pd.DataFrame): Property details
            model_results (dict): Results from prediction models
            user_context (str): User type - 'owner', 'investor', 'agent', or 'tenant'
            user_id (str, optional): Unique identifier for tracking user recommendations
            
        Returns:
            dict: Recommendations tailored to the user context
        """
        if user_context not in self.user_contexts:
            raise ValueError(f"Invalid user context '{user_context}'. Valid contexts are: {', '.join(self.user_contexts)}")
        
        # Convert property_data to dict if it's a DataFrame row
        if isinstance(property_data, pd.DataFrame) and len(property_data) == 1:
            property_data = property_data.iloc[0].to_dict()
        elif isinstance(property_data, pd.Series):
            property_data = property_data.to_dict()
        
        # Generate recommendations based on user context
        if user_context == 'owner':
            recommendations = self._get_owner_recommendations(property_data, model_results)
        elif user_context == 'investor':
            recommendations = self._get_investor_recommendations(property_data, model_results)
        elif user_context == 'agent':
            recommendations = self._get_agent_recommendations(property_data, model_results)
        else:  # tenant
            recommendations = self._get_tenant_recommendations(property_data, model_results)
            
        # Add general information
        recommendations['generated_at'] = datetime.now().isoformat()
        recommendations['user_context'] = user_context
        
        # Store recommendation in history if user_id provided
        if user_id:
            if user_id not in self.recommendation_history:
                self.recommendation_history[user_id] = []
            self.recommendation_history[user_id].append({
                'timestamp': datetime.now(),
                'recommendations': recommendations,
                'property_data': property_data,
                'user_context': user_context
            })
        
        return recommendations
    
    def _get_owner_recommendations(self, property_data, model_results):
        """
        Generate recommendations for property owners.
        
        Property owners care about:
        - Optimal rental price
        - Price trend forecasts
        - Timing for rent adjustments
        - Market comparisons
        
        Parameters:
            property_data (dict): Property details
            model_results (dict): Results from prediction models
            
        Returns:
            dict: Owner-focused recommendations
        """
        recommendations = {
            'summary': '',
            'actions': [],
            'insights': [],
            'comparisons': []
        }
        
        # Check if we have rental price prediction results
        if 'rent_prediction' in model_results:
            rent_results = model_results['rent_prediction']
            current_rent = rent_results.get('current_annual_rental', property_data.get('annual_rental_price', 0))
            optimal_rent = rent_results.get('predicted_annual_rent', 0)
            
            if current_rent > 0 and optimal_rent > 0:
                rent_diff_pct = ((optimal_rent / current_rent) - 1) * 100
                monthly_current = current_rent / 12
                monthly_optimal = optimal_rent / 12
                
                # Determine rent recommendation based on difference
                if abs(rent_diff_pct) < 3:
                    action = {
                        'type': 'maintain',
                        'title': 'Maintain Current Rent',
                        'description': f'Your current rent (${monthly_current:.0f}/month) is well-aligned with the market value (${monthly_optimal:.0f}/month).',
                        'impact': 'LOW'
                    }
                    recommendations['summary'] = 'Your property is optimally priced for the current market.'
                elif rent_diff_pct > 0:
                    # Potential to increase rent
                    increase_pct = min(rent_diff_pct, self.decision_thresholds['rent_increase_limit'])
                    new_monthly = monthly_current * (1 + increase_pct/100)
                    
                    action = {
                        'type': 'increase',
                        'title': 'Consider Rent Increase',
                        'description': f'Market analysis suggests your property is underpriced by {rent_diff_pct:.1f}%.',
                        'recommendation': f'Consider increasing rent from ${monthly_current:.0f} to ${new_monthly:.0f} per month.',
                        'impact': 'HIGH' if rent_diff_pct > 10 else 'MEDIUM'
                    }
                    recommendations['summary'] = f'Your property has rental upside potential of {rent_diff_pct:.1f}%.'
                else:
                    # Potential to decrease rent
                    decrease_pct = min(abs(rent_diff_pct), self.decision_thresholds['rent_decrease_limit'])
                    new_monthly = monthly_current * (1 - decrease_pct/100)
                    
                    action = {
                        'type': 'decrease',
                        'title': 'Consider Rent Adjustment',
                        'description': f'Your current rent (${monthly_current:.0f}/month) is {abs(rent_diff_pct):.1f}% above the optimal market rate.',
                        'recommendation': f'Consider adjusting rent to ${new_monthly:.0f} per month to reduce vacancy risk.',
                        'impact': 'MEDIUM' if abs(rent_diff_pct) > 10 else 'LOW'
                    }
                    recommendations['summary'] = 'Your property may be at risk of extended vacancy periods due to above-market pricing.'
                
                recommendations['actions'].append(action)
                
                # Add market comparison insights
                if 'market_comparisons' in rent_results:
                    comparisons = rent_results['market_comparisons']
                    
                    if 'area' in comparisons:
                        area_data = comparisons['area']
                        area_name = area_data.get('area', property_data.get('area_name', 'your area'))
                        area_diff_pct = area_data.get('difference_percent', 0)
                        
                        area_comparison = {
                            'type': 'area',
                            'title': f'Comparison to {area_name}',
                            'description': f'Your property is priced {abs(area_diff_pct):.1f}% {"above" if area_diff_pct > 0 else "below"} the average for {area_name}.'
                        }
                        recommendations['comparisons'].append(area_comparison)
                    
                    if 'property_type' in comparisons:
                        type_data = comparisons['property_type']
                        prop_type = type_data.get('type', property_data.get('property_type', 'this property type'))
                        type_diff_pct = type_data.get('difference_percent', 0)
                        
                        type_comparison = {
                            'type': 'property_type',
                            'title': f'Comparison to Similar {prop_type} Properties',
                            'description': f'Your property is priced {abs(type_diff_pct):.1f}% {"above" if type_diff_pct > 0 else "below"} the average for similar {prop_type} properties.'
                        }
                        recommendations['comparisons'].append(type_comparison)
            
            # Add seasonal insights if available
            if 'seasonal_trends' in model_results:
                seasonal = model_results['seasonal_trends']
                current_month = datetime.now().month
                
                if 'monthly' in seasonal and str(current_month) in seasonal['monthly']:
                    month_data = seasonal['monthly'][str(current_month)]
                    rent_index = month_data.get('rent_index', 1.0)
                    
                    if rent_index > 1.05:
                        seasonal_insight = {
                            'type': 'seasonal_high',
                            'title': 'Seasonal Peak Period',
                            'description': f'Currently in a high-demand season for rentals. This is a good time to maintain or increase rent.'
                        }
                    elif rent_index < 0.95:
                        seasonal_insight = {
                            'type': 'seasonal_low',
                            'title': 'Seasonal Low Period',
                            'description': f'Currently in a lower-demand season for rentals. Consider incentives rather than decreasing base rent.'
                        }
                    else:
                        seasonal_insight = {
                            'type': 'seasonal_neutral',
                            'title': 'Neutral Seasonal Period',
                            'description': f'Currently in a neutral season for rental demand.'
                        }
                    
                    recommendations['insights'].append(seasonal_insight)
        
        # Add market trend insights if available
        if 'market_trends' in model_results:
            trends = model_results['market_trends']
            
            if 'price_trend' in trends:
                price_trend = trends['price_trend']
                trend_pct = price_trend.get('annual_change_percent', 0)
                
                trend_insight = {
                    'type': 'price_trend',
                    'title': 'Market Price Trend',
                    'description': f'Rental prices in this market have {"increased" if trend_pct > 0 else "decreased"} by {abs(trend_pct):.1f}% in the past year.'
                }
                recommendations['insights'].append(trend_insight)
        
        return recommendations
    
    def _get_investor_recommendations(self, property_data, model_results):
        """
        Generate recommendations for real estate investors.
        
        Investors care about:
        - Investment opportunities
        - ROI and yield metrics
        - Market growth potential
        - Property valuation
        
        Parameters:
            property_data (dict): Property details
            model_results (dict): Results from prediction models
            
        Returns:
            dict: Investor-focused recommendations
        """
        recommendations = {
            'summary': '',
            'actions': [],
            'insights': [],
            'metrics': []
        }
        
        # Check for investment score results
        if 'investment_score' in model_results:
            score = model_results['investment_score']
            overall_score = score.get('overall_score', 0)
            
            # Add investment score metrics
            recommendations['metrics'].append({
                'name': 'Investment Score',
                'value': overall_score,
                'max_value': 100,
                'description': f'Overall investment quality score: {overall_score:.1f}/100'
            })
            
            # Add component scores if available
            for component in ['valuation_score', 'yield_score', 'rental_upside_score', 'location_score']:
                if component in score:
                    component_name = component.replace('_score', '').title()
                    component_value = score[component]
                    recommendations['metrics'].append({
                        'name': component_name,
                        'value': component_value,
                        'description': f'{component_name} score: {component_value:.1f}'
                    })
            
            # Generate summary based on overall score
            if overall_score >= 80:
                recommendations['summary'] = 'Excellent investment opportunity with strong potential returns.'
            elif overall_score >= 60:
                recommendations['summary'] = 'Good investment opportunity with above-average potential returns.'
            elif overall_score >= 40:
                recommendations['summary'] = 'Average investment opportunity with standard market returns expected.'
            else:
                recommendations['summary'] = 'Below average investment opportunity. Consider alternatives for better returns.'
        
        # Check for valuation results
        if 'valuation' in model_results:
            valuation = model_results['valuation']
            current_price = valuation.get('current_value', property_data.get('property_price', 0))
            fair_value = valuation.get('fair_market_value', 0)
            
            if current_price > 0 and fair_value > 0:
                value_diff_pct = valuation.get('value_difference_percent', ((fair_value / current_price) - 1) * 100)
                undervalued = value_diff_pct > 0
                
                if undervalued and value_diff_pct > self.decision_thresholds['undervalue_threshold']:
                    action = {
                        'type': 'buy_opportunity',
                        'title': 'Strong Buy Opportunity',
                        'description': f'Property appears undervalued by {value_diff_pct:.1f}%.',
                        'recommendation': 'Consider purchasing at current price for potential appreciation gains.',
                        'impact': 'HIGH'
                    }
                elif undervalued:
                    action = {
                        'type': 'potential_buy',
                        'title': 'Potential Buy Opportunity',
                        'description': f'Property appears slightly undervalued by {value_diff_pct:.1f}%.',
                        'recommendation': 'Consider purchasing if other investment criteria are met.',
                        'impact': 'MEDIUM'
                    }
                elif value_diff_pct < -self.decision_thresholds['undervalue_threshold']:
                    action = {
                        'type': 'avoid_purchase',
                        'title': 'Consider Alternative Investment',
                        'description': f'Property appears overvalued by {abs(value_diff_pct):.1f}%.',
                        'recommendation': 'Look for better-priced alternatives in this market.',
                        'impact': 'HIGH'
                    }
                else:
                    action = {
                        'type': 'fair_value',
                        'title': 'Fair Market Value',
                        'description': f'Property is priced close to its fair market value.',
                        'recommendation': 'Investment decision should be based on other factors like location and rental yield.',
                        'impact': 'LOW'
                    }
                
                recommendations['actions'].append(action)
                
                # Add valuation metric
                recommendations['metrics'].append({
                    'name': 'Valuation',
                    'value': value_diff_pct,
                    'description': f'{"Undervalued" if undervalued else "Overvalued"} by {abs(value_diff_pct):.1f}%',
                    'current_price': current_price,
                    'fair_value': fair_value
                })
        
        # Check for rental yield information
        rental_yield = property_data.get('rental_yield', 0)
        if rental_yield > 0:
            yield_metric = {
                'name': 'Rental Yield',
                'value': rental_yield,
                'description': f'Annual rental return: {rental_yield:.2f}%'
            }
            recommendations['metrics'].append(yield_metric)
            
            # Add yield-based insights
            if rental_yield > self.decision_thresholds['high_yield']:
                insight = {
                    'type': 'high_yield',
                    'title': 'Strong Rental Yield',
                    'description': f'This property offers an above-average rental yield of {rental_yield:.2f}%.'
                }
                recommendations['insights'].append(insight)
        
        # Check for price-to-rent ratio
        price_to_rent = property_data.get('price_to_rent_ratio', 0)
        if price_to_rent > 0:
            ptr_metric = {
                'name': 'Price-to-Rent Ratio',
                'value': price_to_rent,
                'description': f'Price to annual rent ratio: {price_to_rent:.1f}'
            }
            recommendations['metrics'].append(ptr_metric)
            
            # Add price-to-rent insights
            if price_to_rent < self.decision_thresholds['price_to_rent_good']:
                insight = {
                    'type': 'good_ptr',
                    'title': 'Favorable Price-to-Rent Ratio',
                    'description': f'This property has a favorable price-to-rent ratio of {price_to_rent:.1f}, below the threshold of {self.decision_thresholds["price_to_rent_good"]}.'
                }
                recommendations['insights'].append(insight)
        
        # Add location insights if available
        if 'location_premium' in model_results:
            location_premium = model_results['location_premium']
            premium_pct = location_premium.get('premium_percent', 0)
            
            if premium_pct > self.decision_thresholds['location_premium_threshold']:
                insight = {
                    'type': 'premium_location',
                    'title': 'Premium Location',
                    'description': f'This property is in a premium location with values {premium_pct:.1f}% above the market average.'
                }
                recommendations['insights'].append(insight)
                
            # Add location growth potential if available
            if 'growth_potential' in location_premium:
                growth = location_premium['growth_potential']
                insight = {
                    'type': 'location_growth',
                    'title': 'Location Growth Potential',
                    'description': f'This area has {growth} growth potential based on development plans and market trends.'
                }
                recommendations['insights'].append(insight)
        
        return recommendations
    
    def _get_agent_recommendations(self, property_data, model_results):
        """
        Generate recommendations for real estate agents.
        
        Agents care about:
        - Optimal listing price
        - Marketing strategy
        - Comparable properties
        - Market position
        
        Parameters:
            property_data (dict): Property details
            model_results (dict): Results from prediction models
            
        Returns:
            dict: Agent-focused recommendations
        """
        recommendations = {
            'summary': '',
            'pricing_strategy': {},
            'marketing_points': [],
            'comparable_properties': [],
            'market_position': {}
        }
        
        # Determine pricing strategy based on valuation and rental results
        if 'valuation' in model_results:
            valuation = model_results['valuation']
            current_price = valuation.get('current_value', property_data.get('property_price', 0))
            fair_value = valuation.get('fair_market_value', 0)
            
            if current_price > 0 and fair_value > 0:
                value_diff_pct = valuation.get('value_difference_percent', ((fair_value / current_price) - 1) * 100)
                
                pricing_strategy = {
                    'current_price': current_price,
                    'recommended_price': fair_value,
                    'price_difference_percent': value_diff_pct
                }
                
                if abs(value_diff_pct) < 5:
                    pricing_strategy['strategy'] = 'MARKET_ALIGNED'
                    pricing_strategy['description'] = 'Price is well-aligned with market value. Focus on property features and benefits in marketing.'
                elif value_diff_pct > 0:
                    pricing_strategy['strategy'] = 'UNDERPRICED'
                    pricing_strategy['description'] = f'Property appears underpriced by {value_diff_pct:.1f}%. Consider increasing asking price or highlighting investment potential.'
                else:
                    pricing_strategy['strategy'] = 'PREMIUM_PRICING'
                    pricing_strategy['description'] = f'Property is priced at a {abs(value_diff_pct):.1f}% premium. Ensure marketing emphasizes unique value propositions.'
                
                recommendations['pricing_strategy'] = pricing_strategy
        
        # Generate marketing points based on property attributes and market position
        if 'amenities' in property_data and property_data['amenities']:
            amenities_list = property_data['amenities'] if isinstance(property_data['amenities'], list) else property_data['amenities'].split(',')
            
            premium_amenities = [amenity for amenity in amenities_list if str(amenity).lower() in self.amenity_value_map and self.amenity_value_map[str(amenity).lower()] >= 0.04]
            
            if premium_amenities:
                recommendations['marketing_points'].append({
                    'type': 'premium_amenities',
                    'title': 'Premium Amenities',
                    'description': f'Highlight premium features: {", ".join(premium_amenities)}.'
                })
        
        # Add market position insights
        if 'market_trends' in model_results:
            trends = model_results['market_trends']
            
            recommendations['market_position'] = {
                'market_type': trends.get('market_type', 'BALANCED'),
                'description': trends.get('market_description', 'Current market conditions are balanced between buyers and sellers.')
            }
            
            if trends.get('price_trend', {}).get('direction') == 'up':
                recommendations['marketing_points'].append({
                    'type': 'rising_market',
                    'title': 'Rising Market',
                    'description': 'Emphasize market appreciation potential in marketing materials.'
                })
        
        # Include rental potential for investors if data available
        if 'rent_prediction' in model_results and property_data.get('property_price', 0) > 0:
            rent_results = model_results['rent_prediction']
            predicted_annual_rent = rent_results.get('predicted_annual_rent', 0)
            
            if predicted_annual_rent > 0:
                rental_yield = (predicted_annual_rent / property_data['property_price']) * 100
                
                if rental_yield > self.decision_thresholds['high_yield']:
                    recommendations['marketing_points'].append({
                        'type': 'investment_potential',
                        'title': 'Strong Investment Potential',
                        'description': f'Highlight potential {rental_yield:.1f}% rental yield for investors.'
                    })
        
        # Add comparable properties if available
        if 'comparable_properties' in model_results:
            comps = model_results['comparable_properties']
            recommendations['comparable_properties'] = comps
        
        # Generate summary based on all available data
        if 'pricing_strategy' in recommendations and recommendations['pricing_strategy']:
            strategy = recommendations['pricing_strategy']['strategy']
            
            if strategy == 'MARKET_ALIGNED':
                recommendations['summary'] = 'Property is well-positioned in the market. Focus on unique features and benefits.'
            elif strategy == 'UNDERPRICED':
                recommendations['summary'] = 'Property appears underpriced. Consider adjusting price or emphasizing investment value.'
            else:  # PREMIUM_PRICING
                recommendations['summary'] = 'Property is premium-priced. Highlight distinctive features and justify the premium positioning.'
        
        return recommendations
    
    def _get_tenant_recommendations(self, property_data, model_results):
        """
        Generate recommendations for prospective tenants.
        
        Tenants care about:
        - Fair rental price
        - Value for money
        - Area insights
        - Negotiation points
        
        Parameters:
            property_data (dict): Property details
            model_results (dict): Results from prediction models
            
        Returns:
            dict: Tenant-focused recommendations
        """
        recommendations = {
            'summary': '',
            'rental_value': {},
            'area_insights': [],
            'negotiation_points': []
        }
        
        # Determine rental value assessment
        if 'rent_prediction' in model_results:
            rent_results = model_results['rent_prediction']
            current_rent = rent_results.get('current_annual_rental', property_data.get('annual_rental_price', 0))
            optimal_rent = rent_results.get('predicted_annual_rent', 0)
            
            if current_rent > 0 and optimal_rent > 0:
                rent_diff_pct = ((current_rent / optimal_rent) - 1) * 100
                monthly_current = current_rent / 12
                monthly_optimal = optimal_rent / 12
                
                rental_value = {
                    'current_monthly_rent': monthly_current,
                    'fair_monthly_rent': monthly_optimal,
                    'difference_percent': rent_diff_pct
                }
                
                if abs(rent_diff_pct) < 5:
                    rental_value['assessment'] = 'FAIR_PRICE'
                    rental_value['description'] = 'This property is fairly priced compared to similar properties in the area.'
                elif rent_diff_pct > 0:
                    rental_value['assessment'] = 'OVERPRICED'
                    rental_value['description'] = f'This property appears to be overpriced by {rent_diff_pct:.1f}% compared to similar properties.'
                    
                    # Add negotiation point
                    recommendations['negotiation_points'].append({
                        'type': 'price_negotiation',
                        'title': 'Price Negotiation',
                        'description': f'Consider negotiating a lower rent closer to ${monthly_optimal:.0f} per month based on market rates for similar properties.'
                    })
                else:
                    rental_value['assessment'] = 'GOOD_VALUE'
                    rental_value['description'] = f'This property offers good value, priced {abs(rent_diff_pct):.1f}% below similar properties in the area.'
                
                recommendations['rental_value'] = rental_value
        
        # Add area insights if available
        if 'area_data' in model_results:
            area_data = model_results['area_data']
            
            # Add transportation insights
            if 'transportation' in area_data:
                transport = area_data['transportation']
                recommendations['area_insights'].append({
                    'type': 'transportation',
                    'title': 'Transportation',
                    'description': transport.get('description', 'Transportation information for this area.')
                })
            
            # Add amenities insights
            if 'nearby_amenities' in area_data:
                amenities = area_data['nearby_amenities']
                recommendations['area_insights'].append({
                    'type': 'amenities',
                    'title': 'Nearby Amenities',
                    'description': f"Nearby amenities include: {', '.join(amenities[:5])}."
                })
            
            # Add safety insights
            if 'safety' in area_data:
                safety = area_data['safety']
                recommendations['area_insights'].append({
                    'type': 'safety',
                    'title': 'Neighborhood Safety',
                    'description': safety.get('description', 'Safety information for this area.')
                })
        
        # Add property-specific negotiation points
        if property_data.get('property_age', 0) > 10:
            recommendations['negotiation_points'].append({
                'type': 'maintenance',
                'title': 'Maintenance Considerations',
                'description': 'This is an older property. Consider discussing maintenance responsibilities and history with the landlord.'
            })
        
        if 'amenities' in property_data and property_data['amenities']:
            # Check if property lacks common amenities
            amenities_list = property_data['amenities'] if isinstance(property_data['amenities'], list) else property_data['amenities'].split(',')
            common_amenities = ['parking', 'security', 'gym', 'pool']
            missing_amenities = [amenity for amenity in common_amenities if not any(missing in str(existing).lower() for existing in amenities_list for missing in [amenity])]
            
            if missing_amenities:
                recommendations['negotiation_points'].append({
                    'type': 'missing_amenities',
                    'title': 'Missing Amenities',
                    'description': f'Property lacks some common amenities: {", ".join(missing_amenities)}. Consider this when evaluating the asking price.'
                })
        
        # Generate summary based on rental value
        if 'rental_value' in recommendations and recommendations['rental_value']:
            assessment = recommendations['rental_value']['assessment']
            
            if assessment == 'FAIR_PRICE':
                recommendations['summary'] = 'This property is fairly priced for its features and location.'
            elif assessment == 'OVERPRICED':
                recommendations['summary'] = 'This property appears overpriced compared to market rates. Consider negotiation or exploring alternatives.'
            else:  # GOOD_VALUE
                recommendations['summary'] = 'This property offers good value compared to similar properties in the area.'
        
        return recommendations
    
    def track_recommendation_feedback(self, recommendation_id, user_id, feedback_type, feedback_data=None):
        """
        Track feedback on recommendations for continuous improvement.
        
        Parameters:
            recommendation_id (str): ID of the recommendation
            user_id (str): ID of the user providing feedback
            feedback_type (str): Type of feedback ('accepted', 'rejected', 'modified', etc.)
            feedback_data (dict, optional): Additional feedback details
            
        Returns:
            bool: Whether feedback was successfully recorded
        """
        if user_id not in self.recommendation_history:
            return False
        
        timestamp = datetime.now()
        feedback = {
            'timestamp': timestamp,
            'recommendation_id': recommendation_id,
            'user_id': user_id,
            'feedback_type': feedback_type,
            'feedback_data': feedback_data or {}
        }
        
        # Find the recommendation and add feedback
        for rec in self.recommendation_history[user_id]:
            if rec.get('id') == recommendation_id:
                if 'feedback' not in rec:
                    rec['feedback'] = []
                rec['feedback'].append(feedback)
                return True
        
        return False
    
    def update_decision_thresholds(self, new_thresholds):
        """
        Update the decision thresholds used in recommendations.
        
        Parameters:
            new_thresholds (dict): New threshold values
            
        Returns:
            dict: Updated thresholds
        """
        self.decision_thresholds.update(new_thresholds)
        return self.decision_thresholds