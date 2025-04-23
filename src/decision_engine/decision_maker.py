"""
DecisionMaker: Module for automated decision making based on recommendations

This module provides a decision engine that can automatically make decisions
based on recommendation results, user feedback, and business rules.
"""

from datetime import datetime


class DecisionEngine:
    """
    An automated decision making engine for real estate actions based on
    recommendations and feedback data.
    """
    
    def __init__(self):
        """Initialize the decision engine."""
        self.decision_history = []
        self.pending_decisions = {}
        self.automation_rules = self._init_default_rules()
        self.approval_thresholds = {
            'rent_increase': {
                'auto_approve_pct': 3.0,     # Auto-approve increases up to 3%
                'auto_reject_pct': 15.0      # Auto-reject increases above 15%
            },
            'rent_decrease': {
                'auto_approve_pct': 5.0,     # Auto-approve decreases up to 5%
                'auto_reject_pct': 20.0      # Auto-reject decreases above 20%
            },
            'property_purchase': {
                'auto_approve_value': 50000,  # Auto-approve if undervalued by up to $50k
                'auto_reject_value': -100000  # Auto-reject if overvalued by more than $100k
            }
        }
    
    def _init_default_rules(self):
        """Initialize default automation rules."""
        return {
            'owner': {
                'rent_adjustment': {
                    'enabled': True,
                    'max_auto_increase_pct': 3.0,
                    'max_auto_decrease_pct': 5.0
                },
                'maintenance': {
                    'enabled': True,
                    'max_auto_approve_amount': 500
                }
            },
            'investor': {
                'property_purchase': {
                    'enabled': True,
                    'min_investment_score': 75,
                    'min_rental_yield': 6.0,
                    'max_auto_approve_amount': 0  # Default to manual approval for purchases
                },
                'property_sale': {
                    'enabled': False  # Default to manual approval for sales
                }
            },
            'agent': {
                'listing_creation': {
                    'enabled': True,
                    'requires_photo_verification': True
                },
                'price_adjustment': {
                    'enabled': True,
                    'max_auto_adjust_pct': 3.0
                }
            }
        }
    
    def make_decision(self, recommendation, user_id=None, context_type=None, property_data=None):
        """
        Make or recommend a decision based on recommendation data and automation rules.
        
        Parameters:
            recommendation (dict): Recommendation data
            user_id (str, optional): ID of the user associated with the decision
            context_type (str, optional): User context type
            property_data (dict, optional): Property data
            
        Returns:
            dict: Decision object with status and details
        """
        # Extract key information from recommendation
        if not recommendation:
            raise ValueError("Empty recommendation")
            
        if not context_type and 'user_context' in recommendation:
            context_type = recommendation['user_context']
        
        if not context_type:
            raise ValueError("Context type is required for decision making")
            
        # Identify decision type and key metrics
        decision_type, decision_metrics = self._extract_decision_metrics(recommendation, context_type)
        
        if not decision_type:
            raise ValueError(f"No actionable recommendation found for context: {context_type}")
            
        # Apply automation rules
        decision = self._apply_automation_rules(decision_type, decision_metrics, context_type)
        
        # Add metadata
        decision.update({
            'id': f"decision_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'created_at': datetime.now().isoformat(),
            'user_id': user_id,
            'context_type': context_type,
            'recommendation_id': recommendation.get('id', 'unknown'),
            'property_data': property_data or {},
            'decision_type': decision_type,
            'decision_metrics': decision_metrics
        })
        
        # Store decision
        self.decision_history.append(decision)
        
        # If pending approval, add to pending decisions
        if decision['status'] == 'pending_approval':
            self.pending_decisions[decision['id']] = decision
            
        return decision
    
    def _extract_decision_metrics(self, recommendation, context_type):
        """
        Extract decision type and key metrics from a recommendation.
        
        Parameters:
            recommendation (dict): Recommendation data
            context_type (str): User context type
            
        Returns:
            tuple: (decision_type, metrics_dict)
        """
        decision_type = None
        metrics = {}
        
        if context_type == 'owner':
            # Look for rent adjustment recommendations
            if 'actions' in recommendation and recommendation['actions']:
                action = recommendation['actions'][0]
                if action.get('type') == 'increase':
                    decision_type = 'rent_increase'
                    # Extract increase percentage from description
                    description = action.get('description', '')
                    if '%' in description:
                        try:
                            metrics['increase_pct'] = float(description.split('%')[0].split()[-1])
                        except (ValueError, IndexError):
                            metrics['increase_pct'] = 0
                    metrics['current_rent'] = recommendation.get('current_rent', 0)
                    metrics['recommended_rent'] = recommendation.get('recommended_rent', 0)
                    if 'recommendation' in action:
                        rec_text = action['recommendation']
                        if '$' in rec_text:
                            try:
                                metrics['recommended_rent'] = float(rec_text.split('$')[1].split()[0].replace(',', ''))
                            except (ValueError, IndexError):
                                pass
                elif action.get('type') == 'decrease':
                    decision_type = 'rent_decrease'
                    # Extract decrease percentage from description
                    description = action.get('description', '')
                    if '%' in description:
                        try:
                            metrics['decrease_pct'] = float(description.split('%')[0].split()[-1])
                        except (ValueError, IndexError):
                            metrics['decrease_pct'] = 0
                    metrics['current_rent'] = recommendation.get('current_rent', 0)
                    metrics['recommended_rent'] = recommendation.get('recommended_rent', 0)
                    if 'recommendation' in action:
                        rec_text = action['recommendation']
                        if '$' in rec_text:
                            try:
                                metrics['recommended_rent'] = float(rec_text.split('$')[1].split()[0].replace(',', ''))
                            except (ValueError, IndexError):
                                pass
                    
        elif context_type == 'investor':
            # Look for property purchase recommendations
            if 'metrics' in recommendation:
                investment_metrics = recommendation['metrics']
                investment_score = next((m for m in investment_metrics if m.get('name') == 'Investment Score'), {})
                rental_yield = next((m for m in investment_metrics if m.get('name') == 'Rental Yield'), {})
                valuation = next((m for m in investment_metrics if m.get('name') == 'Valuation'), {})
                
                if investment_score and rental_yield and valuation:
                    decision_type = 'property_purchase'
                    metrics['investment_score'] = investment_score.get('value', 0)
                    metrics['rental_yield'] = rental_yield.get('value', 0)
                    metrics['valuation_diff'] = valuation.get('value', 0)
                    metrics['current_price'] = valuation.get('current_price', 0)
                    metrics['fair_value'] = valuation.get('fair_value', 0)
        
        elif context_type == 'agent':
            # Look for price adjustment recommendations
            if 'pricing_strategy' in recommendation:
                strategy = recommendation['pricing_strategy']
                if strategy.get('strategy') in ['UNDERPRICED', 'PREMIUM_PRICING']:
                    decision_type = 'price_adjustment'
                    metrics['price_diff_pct'] = strategy.get('price_difference_percent', 0)
                    metrics['current_price'] = strategy.get('current_price', 0)
                    metrics['recommended_price'] = strategy.get('recommended_price', 0)
        
        return decision_type, metrics
    
    def _apply_automation_rules(self, decision_type, metrics, context_type):
        """
        Apply automation rules to determine decision status.
        
        Parameters:
            decision_type (str): Type of decision
            metrics (dict): Decision metrics
            context_type (str): User context type
            
        Returns:
            dict: Decision with status and justification
        """
        decision = {
            'status': 'pending_approval',  # Default status
            'automated': False,
            'justification': 'Manual review required.',
            'approval_level': 'manager'
        }
        
        # Apply context-specific rules
        if context_type == 'owner':
            if decision_type == 'rent_increase':
                increase_pct = metrics.get('increase_pct', 0)
                
                # Check if auto-approval is enabled
                auto_enabled = self.automation_rules.get('owner', {}).get('rent_adjustment', {}).get('enabled', False)
                max_auto_pct = self.automation_rules.get('owner', {}).get('rent_adjustment', {}).get('max_auto_increase_pct', 0)
                
                if auto_enabled and increase_pct <= max_auto_pct:
                    decision['status'] = 'approved'
                    decision['automated'] = True
                    decision['justification'] = f'Automatically approved rent increase of {increase_pct:.1f}% (within {max_auto_pct}% threshold).'
                elif increase_pct > self.approval_thresholds['rent_increase']['auto_reject_pct']:
                    decision['status'] = 'rejected'
                    decision['automated'] = True
                    decision['justification'] = f'Automatically rejected excessive rent increase of {increase_pct:.1f}% (above {self.approval_thresholds["rent_increase"]["auto_reject_pct"]}% threshold).'
                else:
                    decision['justification'] = f'Rent increase of {increase_pct:.1f}% requires manual review.'
                    
            elif decision_type == 'rent_decrease':
                decrease_pct = metrics.get('decrease_pct', 0)
                
                # Check if auto-approval is enabled
                auto_enabled = self.automation_rules.get('owner', {}).get('rent_adjustment', {}).get('enabled', False)
                max_auto_pct = self.automation_rules.get('owner', {}).get('rent_adjustment', {}).get('max_auto_decrease_pct', 0)
                
                if auto_enabled and decrease_pct <= max_auto_pct:
                    decision['status'] = 'approved'
                    decision['automated'] = True
                    decision['justification'] = f'Automatically approved rent decrease of {decrease_pct:.1f}% (within {max_auto_pct}% threshold).'
                elif decrease_pct > self.approval_thresholds['rent_decrease']['auto_reject_pct']:
                    decision['status'] = 'rejected'
                    decision['automated'] = True
                    decision['justification'] = f'Automatically rejected excessive rent decrease of {decrease_pct:.1f}% (above {self.approval_thresholds["rent_decrease"]["auto_reject_pct"]}% threshold).'
                else:
                    decision['justification'] = f'Rent decrease of {decrease_pct:.1f}% requires manual review.'
                    
        elif context_type == 'investor':
            if decision_type == 'property_purchase':
                investment_score = metrics.get('investment_score', 0)
                rental_yield = metrics.get('rental_yield', 0)
                valuation_diff = metrics.get('valuation_diff', 0)
                
                # Check if auto-approval is enabled
                auto_enabled = self.automation_rules.get('investor', {}).get('property_purchase', {}).get('enabled', False)
                min_score = self.automation_rules.get('investor', {}).get('property_purchase', {}).get('min_investment_score', 0)
                min_yield = self.automation_rules.get('investor', {}).get('property_purchase', {}).get('min_rental_yield', 0)
                
                if auto_enabled and investment_score >= min_score and rental_yield >= min_yield and valuation_diff > 0:
                    # Check if valuation difference is within auto-approve threshold
                    max_auto_value = self.automation_rules.get('investor', {}).get('property_purchase', {}).get('max_auto_approve_amount', 0)
                    current_price = metrics.get('current_price', 0)
                    
                    if max_auto_value > 0 and current_price <= max_auto_value:
                        decision['status'] = 'approved'
                        decision['automated'] = True
                        decision['justification'] = (
                            f'Automatically approved property purchase. '
                            f'Score: {investment_score:.1f}/{min_score}, '
                            f'Yield: {rental_yield:.2f}%/{min_yield}%, '
                            f'Undervalued by {valuation_diff:.1f}%.'
                        )
                    else:
                        decision['justification'] = 'Property meets investment criteria but price exceeds auto-approval threshold.'
                        decision['approval_level'] = 'director'
                elif valuation_diff < self.approval_thresholds['property_purchase']['auto_reject_value']:
                    decision['status'] = 'rejected'
                    decision['automated'] = True
                    decision['justification'] = f'Automatically rejected overvalued property purchase (over threshold).'
                else:
                    conditions = []
                    if investment_score < min_score:
                        conditions.append(f'Investment score ({investment_score:.1f}) below threshold ({min_score})')
                    if rental_yield < min_yield:
                        conditions.append(f'Rental yield ({rental_yield:.2f}%) below threshold ({min_yield}%)')
                    if valuation_diff <= 0:
                        conditions.append('Property not undervalued')
                        
                    decision['justification'] = 'Manual review required: ' + '; '.join(conditions)
                    
        elif context_type == 'agent':
            if decision_type == 'price_adjustment':
                price_diff_pct = metrics.get('price_diff_pct', 0)
                
                # Check if auto-approval is enabled
                auto_enabled = self.automation_rules.get('agent', {}).get('price_adjustment', {}).get('enabled', False)
                max_auto_pct = self.automation_rules.get('agent', {}).get('price_adjustment', {}).get('max_auto_adjust_pct', 0)
                
                if auto_enabled and abs(price_diff_pct) <= max_auto_pct:
                    decision['status'] = 'approved'
                    decision['automated'] = True
                    decision['justification'] = f'Automatically approved price adjustment of {abs(price_diff_pct):.1f}% (within {max_auto_pct}% threshold).'
                else:
                    decision['justification'] = f'Price adjustment of {abs(price_diff_pct):.1f}% requires manual review.'
        
        return decision
    
    def update_decision_status(self, decision_id, new_status, approver_id=None, notes=None):
        """
        Update the status of a pending decision.
        
        Parameters:
            decision_id (str): ID of the decision to update
            new_status (str): New status ('approved', 'rejected', 'on_hold')
            approver_id (str, optional): ID of the user approving/rejecting
            notes (str, optional): Additional notes
            
        Returns:
            dict: Updated decision object or None if not found
        """
        # Find the decision in pending decisions
        if decision_id not in self.pending_decisions:
            # Check in history
            for decision in self.decision_history:
                if decision['id'] == decision_id:
                    if decision['status'] not in ['pending_approval', 'on_hold']:
                        return None  # Can't update already approved/rejected decisions
                    break
            else:
                return None  # Decision not found
                
        decision = self.pending_decisions[decision_id]
        
        # Update status
        decision['status'] = new_status
        decision['updated_at'] = datetime.now().isoformat()
        decision['approver_id'] = approver_id
        
        if notes:
            decision['notes'] = notes
            
        # Remove from pending if no longer pending
        if new_status not in ['pending_approval', 'on_hold']:
            self.pending_decisions.pop(decision_id, None)
            
        return decision
    
    def get_pending_decisions(self, approval_level=None, user_id=None):
        """
        Get pending decisions that require approval.
        
        Parameters:
            approval_level (str, optional): Filter by approval level
            user_id (str, optional): Filter by user ID
            
        Returns:
            list: Pending decisions
        """
        pending = list(self.pending_decisions.values())
        
        if approval_level:
            pending = [d for d in pending if d.get('approval_level') == approval_level]
            
        if user_id:
            pending = [d for d in pending if d.get('user_id') == user_id]
            
        return pending
    
    def update_automation_rules(self, context_type, rule_category, rule_updates):
        """
        Update automation rules for a specific context type and category.
        
        Parameters:
            context_type (str): User context type
            rule_category (str): Category of rules to update
            rule_updates (dict): Updates to apply
            
        Returns:
            dict: Updated automation rules
        """
        if context_type not in self.automation_rules:
            self.automation_rules[context_type] = {}
            
        if rule_category not in self.automation_rules[context_type]:
            self.automation_rules[context_type][rule_category] = {}
            
        self.automation_rules[context_type][rule_category].update(rule_updates)
        return self.automation_rules[context_type][rule_category]
    
    def get_decision_statistics(self, timeframe=None):
        """
        Get statistics on automated vs. manual decisions.
        
        Parameters:
            timeframe (str, optional): Timeframe for statistics ('day', 'week', 'month')
            
        Returns:
            dict: Decision statistics
        """
        stats = {
            'total': len(self.decision_history),
            'approved': 0,
            'rejected': 0,
            'pending': len(self.pending_decisions),
            'automated': 0,
            'manual': 0,
            'by_type': {},
            'by_context': {}
        }
        
        # Filter decisions by timeframe if specified
        decisions = self.decision_history
        
        if timeframe:
            now = datetime.now()
            if timeframe == 'day':
                cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif timeframe == 'week':
                cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
                cutoff = cutoff.replace(day=cutoff.day - cutoff.weekday())
            elif timeframe == 'month':
                cutoff = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                raise ValueError(f"Invalid timeframe: {timeframe}")
            
            decisions = [d for d in decisions if datetime.fromisoformat(d['created_at']) >= cutoff]
            
        # Calculate statistics
        for decision in decisions:
            status = decision['status']
            
            if status == 'approved':
                stats['approved'] += 1
            elif status == 'rejected':
                stats['rejected'] += 1
                
            if decision.get('automated', False):
                stats['automated'] += 1
            else:
                stats['manual'] += 1
                
            # Count by decision type
            decision_type = decision.get('decision_type', 'unknown')
            if decision_type not in stats['by_type']:
                stats['by_type'][decision_type] = 0
            stats['by_type'][decision_type] += 1
            
            # Count by context type
            context_type = decision.get('context_type', 'unknown')
            if context_type not in stats['by_context']:
                stats['by_context'][context_type] = 0
            stats['by_context'][context_type] += 1
            
        return stats