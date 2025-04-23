"""
UserContext: Module for managing user context in recommendations

This module handles tracking and managing user context for the recommendation engine,
enabling personalized and contextualized recommendations.
"""

from datetime import datetime


class UserContextManager:
    """
    Manages user contexts and preferences for recommendation personalization.
    
    This class tracks user information, preferences, and interaction history
    to enable more relevant and personalized recommendations.
    """
    
    def __init__(self):
        """Initialize the user context manager."""
        self.users = {}
        self.context_types = ['owner', 'investor', 'agent', 'tenant']
        self.default_context = 'investor'
    
    def register_user(self, user_id, context_type=None, preferences=None):
        """
        Register a new user with the context manager.
        
        Parameters:
            user_id (str): Unique identifier for the user
            context_type (str, optional): User type ('owner', 'investor', 'agent', 'tenant')
            preferences (dict, optional): User preferences
            
        Returns:
            dict: User profile
        """
        if context_type and context_type not in self.context_types:
            raise ValueError(f"Invalid context type '{context_type}'. Valid types are: {', '.join(self.context_types)}")
            
        # Create new user profile
        user_profile = {
            'user_id': user_id,
            'context_type': context_type or self.default_context,
            'preferences': preferences or {},
            'created_at': datetime.now(),
            'last_active': datetime.now(),
            'interaction_history': [],
            'notification_preferences': {
                'email': True,
                'push': True,
                'frequency': 'daily'
            }
        }
        
        self.users[user_id] = user_profile
        return user_profile
    
    def get_user_context(self, user_id):
        """
        Get a user's context information.
        
        Parameters:
            user_id (str): Unique identifier for the user
            
        Returns:
            dict: User context information or None if user not found
        """
        user = self.users.get(user_id)
        if not user:
            return None
            
        # Update last active timestamp
        user['last_active'] = datetime.now()
        
        return {
            'user_id': user['user_id'],
            'context_type': user['context_type'],
            'preferences': user['preferences']
        }
    
    def update_user_context(self, user_id, context_type=None, preferences=None):
        """
        Update a user's context information.
        
        Parameters:
            user_id (str): Unique identifier for the user
            context_type (str, optional): New user type
            preferences (dict, optional): Updated preferences
            
        Returns:
            dict: Updated user context or None if user not found
        """
        if user_id not in self.users:
            return None
            
        if context_type:
            if context_type not in self.context_types:
                raise ValueError(f"Invalid context type '{context_type}'. Valid types are: {', '.join(self.context_types)}")
            self.users[user_id]['context_type'] = context_type
            
        if preferences:
            # Update preferences (don't overwrite existing unless specified)
            self.users[user_id]['preferences'].update(preferences)
            
        self.users[user_id]['last_active'] = datetime.now()
        
        # Return updated context
        return self.get_user_context(user_id)
    
    def record_interaction(self, user_id, interaction_type, data=None):
        """
        Record a user interaction for context refinement.
        
        Parameters:
            user_id (str): Unique identifier for the user
            interaction_type (str): Type of interaction (e.g., 'view_property', 'follow_recommendation')
            data (dict, optional): Interaction data
            
        Returns:
            bool: Whether the interaction was recorded successfully
        """
        if user_id not in self.users:
            return False
            
        interaction = {
            'timestamp': datetime.now(),
            'type': interaction_type,
            'data': data or {}
        }
        
        self.users[user_id]['interaction_history'].append(interaction)
        self.users[user_id]['last_active'] = datetime.now()
        return True
    
    def set_notification_preferences(self, user_id, preferences):
        """
        Set notification preferences for a user.
        
        Parameters:
            user_id (str): Unique identifier for the user
            preferences (dict): Notification preferences
            
        Returns:
            dict: Updated notification preferences or None if user not found
        """
        if user_id not in self.users:
            return None
            
        self.users[user_id]['notification_preferences'].update(preferences)
        return self.users[user_id]['notification_preferences']
    
    def get_users_for_notification(self, notification_type):
        """
        Get users who should receive a specific type of notification.
        
        Parameters:
            notification_type (str): Type of notification ('market_update', 'price_alert', etc.)
            
        Returns:
            list: List of user IDs who should receive this notification
        """
        eligible_users = []
        
        for user_id, user in self.users.items():
            # Skip users who have disabled notifications
            if not user['notification_preferences'].get('email') and not user['notification_preferences'].get('push'):
                continue
                
            # Check if user is interested in this notification type
            if notification_type in user['notification_preferences'].get('types', ['all']):
                eligible_users.append(user_id)
                
        return eligible_users
    
    def get_recommendation_context(self, user_id, property_data=None):
        """
        Get complete context for recommendation generation.
        
        Parameters:
            user_id (str): Unique identifier for the user
            property_data (dict, optional): Property data being analyzed
            
        Returns:
            dict: Complete context for recommendation generation
        """
        if user_id not in self.users:
            # Return default context if user not found
            return {
                'context_type': self.default_context,
                'preferences': {},
                'property_data': property_data or {}
            }
        
        user = self.users[user_id]
        
        # Update last active timestamp
        user['last_active'] = datetime.now()
        
        # Get recent interactions
        recent_interactions = user['interaction_history'][-10:] if user['interaction_history'] else []
        
        return {
            'context_type': user['context_type'],
            'preferences': user['preferences'],
            'property_data': property_data or {},
            'recent_interactions': recent_interactions
        }