"""
AI-Powered Persona Detection System
Analyzes conversation patterns to identify user personas and adapt responses accordingly.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

class PersonaType(Enum):
    SALES_MANAGER = "sales_manager"
    PRODUCT_ANALYST = "product_analyst"
    CUSTOMER_SUCCESS = "customer_success"
    EXECUTIVE = "executive"
    UNKNOWN = "unknown"

@dataclass
class PersonaProfile:
    """Defines characteristics and preferences for each persona"""
    persona_type: PersonaType
    keywords: List[str]
    priorities: List[str]
    response_style: str
    typical_queries: List[str]
    quick_actions: List[str]
    kpi_focus: List[str]

class PersonaDetector:
    """Detects user persona from conversation patterns and query content"""
    
    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.persona_profiles = self._initialize_personas()
        self.conversation_history = []
        
    def _initialize_personas(self) -> Dict[PersonaType, PersonaProfile]:
        """Initialize persona profiles with characteristics"""
        return {
            PersonaType.SALES_MANAGER: PersonaProfile(
                persona_type=PersonaType.SALES_MANAGER,
                keywords=["sales", "revenue", "target", "quota", "commission", "pipeline", 
                         "territory", "performance", "deals", "close rate", "forecast"],
                priorities=["Revenue tracking", "Team performance", "Territory analysis", 
                          "Sales forecasting", "Customer acquisition"],
                response_style="Actionable insights with clear metrics and trends",
                typical_queries=[
                    "What are our top performing territories?",
                    "Which sales reps are meeting their quotas?",
                    "What's our monthly revenue trend?",
                    "Show me the sales pipeline by region"
                ],
                quick_actions=[
                    "View team performance dashboard",
                    "Generate territory analysis report",
                    "Export sales pipeline data",
                    "Create forecast presentation"
                ],
                kpi_focus=["Total Revenue", "Sales Growth %", "Close Rate", "Average Deal Size"]
            ),
            
            PersonaType.PRODUCT_ANALYST: PersonaProfile(
                persona_type=PersonaType.PRODUCT_ANALYST,
                keywords=["product", "category", "inventory", "margin", "cost", "profit",
                         "performance", "analytics", "trends", "subcategory", "model"],
                priorities=["Product performance", "Margin analysis", "Inventory optimization",
                          "Product trends", "Profitability insights"],
                response_style="Detailed analytical breakdowns with statistical insights",
                typical_queries=[
                    "Which products have the highest margins?",
                    "What's the performance trend for mountain bikes?",
                    "Show me inventory levels by category",
                    "Compare product profitability across segments"
                ],
                quick_actions=[
                    "Generate product performance report",
                    "Analyze margin trends",
                    "Export inventory analysis",
                    "Create category comparison chart"
                ],
                kpi_focus=["Gross Margin %", "Product Revenue", "Inventory Turnover", "Profit per Unit"]
            ),
            
            PersonaType.CUSTOMER_SUCCESS: PersonaProfile(
                persona_type=PersonaType.CUSTOMER_SUCCESS,
                keywords=["customer", "retention", "satisfaction", "support", "churn",
                         "loyalty", "engagement", "feedback", "relationship", "lifetime value"],
                priorities=["Customer satisfaction", "Retention analysis", "Support metrics",
                          "Customer lifetime value", "Churn prevention"],
                response_style="Customer-centric insights with relationship focus",
                typical_queries=[
                    "Who are our top customers by value?",
                    "What's our customer retention rate?",
                    "Show me recent customer purchase patterns",
                    "Which customers need attention?"
                ],
                quick_actions=[
                    "View customer health dashboard",
                    "Generate retention analysis",
                    "Export customer segmentation",
                    "Create customer success report"
                ],
                kpi_focus=["Customer Lifetime Value", "Retention Rate", "Customer Satisfaction", "Repeat Purchase Rate"]
            ),
            
            PersonaType.EXECUTIVE: PersonaProfile(
                persona_type=PersonaType.EXECUTIVE,
                keywords=["total", "overall", "company", "strategic", "growth", "overview",
                         "summary", "high-level", "executive", "board", "quarterly", "annual"],
                priorities=["Strategic insights", "High-level metrics", "Growth trends",
                          "Executive summaries", "Board reporting"],
                response_style="Executive summaries with key takeaways and strategic implications",
                typical_queries=[
                    "Give me a high-level overview of our performance",
                    "What are the key trends this quarter?",
                    "Show me our growth metrics",
                    "Prepare an executive summary for the board"
                ],
                quick_actions=[
                    "Generate executive dashboard",
                    "Create board presentation",
                    "Export quarterly summary",
                    "View strategic metrics"
                ],
                kpi_focus=["Total Revenue", "Growth Rate", "Market Share", "ROI"]
            )
        }
    
    def detect_persona(self, query: str, conversation_context: List[str] = None) -> Tuple[PersonaType, float]:
        """
        Detect user persona based on query content and conversation history
        Returns: (detected_persona, confidence_score)
        """
        # Store conversation for pattern analysis
        self.conversation_history.append(query)
        
        # Method 1: Keyword-based detection
        keyword_scores = self._analyze_keywords(query)
        
        # Method 2: LLM-based semantic analysis
        semantic_score = self._semantic_analysis(query, conversation_context)
        
        # Method 3: Pattern analysis from conversation history
        pattern_score = self._analyze_conversation_patterns()
        
        # Combine scores with weights
        combined_scores = {}
        for persona in PersonaType:
            if persona == PersonaType.UNKNOWN:
                continue
            combined_scores[persona] = (
                keyword_scores.get(persona, 0) * 0.4 +
                semantic_score.get(persona, 0) * 0.4 +
                pattern_score.get(persona, 0) * 0.2
            )
        
        # Find the highest scoring persona
        if not combined_scores:
            return PersonaType.UNKNOWN, 0.0
            
        detected_persona = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[detected_persona]
        
        # If confidence is too low, return unknown
        if confidence < 0.3:
            return PersonaType.UNKNOWN, confidence
            
        return detected_persona, confidence
    
    def _analyze_keywords(self, query: str) -> Dict[PersonaType, float]:
        """Analyze query for persona-specific keywords"""
        query_lower = query.lower()
        scores = {}
        
        for persona_type, profile in self.persona_profiles.items():
            keyword_matches = sum(1 for keyword in profile.keywords if keyword in query_lower)
            scores[persona_type] = keyword_matches / len(profile.keywords) if profile.keywords else 0
            
        return scores
    
    def _semantic_analysis(self, query: str, context: List[str] = None) -> Dict[PersonaType, float]:
        """Use LLM for semantic analysis of the query"""
        prompt_template = """
        Analyze the following business query and determine which role/persona is most likely asking it.
        
        Query: "{query}"
        Context: {context}
        
        Personas to consider:
        1. Sales Manager - focused on sales performance, revenue, territories, team metrics
        2. Product Analyst - focused on product performance, margins, inventory, analytics
        3. Customer Success - focused on customer satisfaction, retention, support, relationships
        4. Executive - focused on high-level strategy, overall performance, growth trends
        
        Rate each persona from 0-1 based on how likely they are to ask this query.
        Respond in JSON format:
        {{
            "sales_manager": 0.0-1.0,
            "product_analyst": 0.0-1.0,
            "customer_success": 0.0-1.0,
            "executive": 0.0-1.0
        }}
        """
        
        try:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["query", "context"]
            )
            
            chain = LLMChain(prompt=prompt, llm=self.llm_model)
            context_str = " | ".join(context[-3:]) if context else "No previous context"
            
            response = chain.run(query=query, context=context_str)
            
            # Parse JSON response (simplified - would need better error handling)
            import json
            scores_dict = json.loads(response.strip())
            
            # Convert to PersonaType enum keys
            semantic_scores = {}
            for persona_type in PersonaType:
                if persona_type == PersonaType.UNKNOWN:
                    continue
                key = persona_type.value
                semantic_scores[persona_type] = scores_dict.get(key, 0.0)
                
            return semantic_scores
            
        except Exception as e:
            # Fallback to empty scores if LLM analysis fails
            print(f"Semantic analysis failed: {e}")
            return {persona: 0.0 for persona in PersonaType if persona != PersonaType.UNKNOWN}
    
    def _analyze_conversation_patterns(self) -> Dict[PersonaType, float]:
        """Analyze conversation history for persona patterns"""
        if len(self.conversation_history) < 2:
            return {persona: 0.0 for persona in PersonaType if persona != PersonaType.UNKNOWN}
        
        # Analyze recent conversation for consistent patterns
        recent_queries = self.conversation_history[-5:]  # Last 5 queries
        pattern_scores = {persona: 0.0 for persona in PersonaType if persona != PersonaType.UNKNOWN}
        
        for query in recent_queries:
            query_scores = self._analyze_keywords(query)
            for persona, score in query_scores.items():
                pattern_scores[persona] += score
        
        # Normalize by number of queries
        for persona in pattern_scores:
            pattern_scores[persona] /= len(recent_queries)
            
        return pattern_scores
    
    def get_persona_profile(self, persona_type: PersonaType) -> PersonaProfile:
        """Get the profile for a specific persona"""
        return self.persona_profiles.get(persona_type, None)
    
    def get_response_style(self, persona_type: PersonaType) -> str:
        """Get the appropriate response style for a persona"""
        profile = self.get_persona_profile(persona_type)
        return profile.response_style if profile else "Standard analytical response"
    
    def get_quick_actions(self, persona_type: PersonaType) -> List[str]:
        """Get relevant quick actions for a persona"""
        profile = self.get_persona_profile(persona_type)
        return profile.quick_actions if profile else []
    
    def get_kpi_focus(self, persona_type: PersonaType) -> List[str]:
        """Get the KPIs this persona cares about most"""
        profile = self.get_persona_profile(persona_type)
        return profile.kpi_focus if profile else []

# Example usage
if __name__ == "__main__":
    # Initialize with your existing LLM model
    llm_model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    detector = PersonaDetector(llm_model)
    
    # Test queries
    test_queries = [
        "What's our monthly sales performance by territory?",
        "Which products have the highest profit margins?",
        "Show me our top customers and their purchase history",
        "Give me an executive summary of Q4 performance"
    ]
    
    for query in test_queries:
        persona, confidence = detector.detect_persona(query)
        print(f"Query: {query}")
        print(f"Detected Persona: {persona.value}, Confidence: {confidence:.2f}")
        print(f"Response Style: {detector.get_response_style(persona)}")
        print(f"Quick Actions: {detector.get_quick_actions(persona)}")
        print("-" * 80)
