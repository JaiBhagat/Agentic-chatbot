"""
Persona-Aware Query Processing System
Modifies SQL generation and response formatting based on detected user persona
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
import pandas as pd
import json

from persona_detector import PersonaType, PersonaDetector

@dataclass
class PersonaQueryContext:
    """Context for persona-specific query processing"""
    persona_type: PersonaType
    confidence: float
    original_query: str
    enhanced_query: str
    preferred_metrics: List[str]
    visualization_type: str
    additional_context: Dict[str, Any]

class PersonaQueryProcessor:
    """Processes queries with persona-specific adaptations"""
    
    def __init__(self, llm_model, persona_detector: PersonaDetector):
        self.llm_model = llm_model
        self.persona_detector = persona_detector
        self.persona_templates = self._initialize_persona_templates()
        
    def _initialize_persona_templates(self) -> Dict[PersonaType, Dict[str, str]]:
        """Initialize persona-specific prompt templates"""
        return {
            PersonaType.SALES_MANAGER: {
                "sql_enhancement": """
                As a Sales Manager, you need actionable insights about sales performance.
                Original query: {original_query}
                
                Enhance this query to include:
                - Sales team performance metrics (individual and territory-based)
                - Revenue trends and comparisons
                - Target vs actual performance
                - Pipeline and forecasting data where relevant
                - Time-based analysis (monthly, quarterly trends)
                
                Enhanced query focus: Revenue, territory performance, sales team metrics
                """,
                
                "response_format": """
                Format the response for a Sales Manager:
                - Lead with key revenue/performance numbers
                - Include percentage changes and trends
                - Highlight top/bottom performers
                - Provide actionable insights for sales strategy
                - Use business-friendly language with clear metrics
                
                Data: {data_summary}
                Query context: {query_context}
                
                Sales Manager Response:
                """,
                
                "additional_metrics": [
                    "SELECT employee.FullName, employee.Territory, SUM(sales.LineTotal) as revenue, COUNT(DISTINCT sales.CustomerID) as customers_served",
                    "SELECT DATE_FORMAT(sales.OrderDate, '%Y-%m') as month, SUM(sales.LineTotal) as monthly_revenue",
                    "SELECT employee.Territory, AVG(sales.LineTotal) as avg_deal_size, COUNT(*) as deal_count"
                ]
            },
            
            PersonaType.PRODUCT_ANALYST: {
                "sql_enhancement": """
                As a Product Analyst, you need detailed product performance and profitability insights.
                Original query: {original_query}
                
                Enhance this query to include:
                - Product profitability analysis (revenue, cost, margin)
                - Category and subcategory performance
                - Inventory and sales velocity metrics
                - Product lifecycle and trend analysis
                - Comparative analysis across product lines
                
                Enhanced query focus: Product margins, category performance, inventory analytics
                """,
                
                "response_format": """
                Format the response for a Product Analyst:
                - Start with key profitability metrics
                - Include detailed breakdowns by category/subcategory
                - Show margin analysis and cost structures
                - Provide statistical insights and trends
                - Use analytical language with precise calculations
                
                Data: {data_summary}
                Query context: {query_context}
                
                Product Analyst Response:
                """,
                
                "additional_metrics": [
                    "SELECT products.ProductName, SUM(sales.LineTotal) - SUM(sales.OrderQty * products.StandardCost) as profit_margin",
                    "SELECT productcategory.CategoryName, AVG((sales.LineTotal - sales.OrderQty * products.StandardCost) / sales.LineTotal) as avg_margin_pct",
                    "SELECT products.ProductName, SUM(sales.OrderQty) as units_sold, products.ListPrice, products.StandardCost"
                ]
            },
            
            PersonaType.CUSTOMER_SUCCESS: {
                "sql_enhancement": """
                As a Customer Success Lead, you need insights about customer relationships and satisfaction.
                Original query: {original_query}
                
                Enhance this query to include:
                - Customer lifetime value and purchase patterns
                - Customer segmentation and behavior analysis
                - Retention and repeat purchase metrics
                - Customer relationship health indicators
                - Support and satisfaction relevant data
                
                Enhanced query focus: Customer relationships, retention, lifetime value
                """,
                
                "response_format": """
                Format the response for a Customer Success Lead:
                - Emphasize customer relationship insights
                - Include customer segmentation and behavior patterns
                - Highlight retention and satisfaction indicators
                - Provide customer-centric recommendations
                - Use relationship-focused language
                
                Data: {data_summary}
                Query context: {query_context}
                
                Customer Success Response:
                """,
                
                "additional_metrics": [
                    "SELECT customers.FullName, COUNT(DISTINCT sales.SalesOrderID) as order_count, SUM(sales.LineTotal) as lifetime_value",
                    "SELECT customers.CustomerID, MIN(sales.OrderDate) as first_purchase, MAX(sales.OrderDate) as last_purchase",
                    "SELECT DATEDIFF(month, MIN(sales.OrderDate), MAX(sales.OrderDate)) as customer_tenure_months, customers.FullName"
                ]
            },
            
            PersonaType.EXECUTIVE: {
                "sql_enhancement": """
                As an Executive, you need high-level strategic insights and key performance indicators.
                Original query: {original_query}
                
                Enhance this query to include:
                - High-level business metrics and KPIs
                - Strategic trends and growth indicators
                - Executive summary relevant data points
                - Comparative performance across key dimensions
                - Board-ready insights and strategic implications
                
                Enhanced query focus: Strategic KPIs, growth trends, executive insights
                """,
                
                "response_format": """
                Format the response for an Executive:
                - Start with the key strategic takeaway
                - Present high-level metrics and trends
                - Include growth percentages and strategic indicators
                - Provide strategic implications and recommendations
                - Use executive-level language with clear bottom line
                
                Data: {data_summary}
                Query context: {query_context}
                
                Executive Summary:
                """,
                
                "additional_metrics": [
                    "SELECT SUM(sales.LineTotal) as total_revenue, COUNT(DISTINCT customers.CustomerID) as total_customers",
                    "SELECT DATE_FORMAT(sales.OrderDate, '%Y') as year, SUM(sales.LineTotal) as annual_revenue",
                    "SELECT COUNT(DISTINCT employee.EmployeeID) as sales_team_size, AVG(sales.LineTotal) as avg_transaction_value"
                ]
            }
        }
    
    def process_query(self, original_query: str, conversation_context: List[str] = None) -> PersonaQueryContext:
        """Process query with persona-specific enhancements"""
        
        # Detect persona
        persona_type, confidence = self.persona_detector.detect_persona(
            original_query, conversation_context
        )
        
        # Enhance query based on persona
        enhanced_query = self._enhance_query_for_persona(original_query, persona_type)
        
        # Get persona-specific preferences
        preferred_metrics = self.persona_detector.get_kpi_focus(persona_type)
        visualization_type = self._get_preferred_visualization(persona_type)
        
        return PersonaQueryContext(
            persona_type=persona_type,
            confidence=confidence,
            original_query=original_query,
            enhanced_query=enhanced_query,
            preferred_metrics=preferred_metrics,
            visualization_type=visualization_type,
            additional_context={}
        )
    
    def _enhance_query_for_persona(self, original_query: str, persona_type: PersonaType) -> str:
        """Enhance the original query based on persona needs"""
        
        if persona_type == PersonaType.UNKNOWN:
            return original_query
            
        enhancement_template = self.persona_templates[persona_type]["sql_enhancement"]
        
        try:
            prompt = PromptTemplate(
                template=enhancement_template,
                input_variables=["original_query"]
            )
            
            chain = LLMChain(prompt=prompt, llm=self.llm_model)
            enhanced_query = chain.run(original_query=original_query)
            
            return enhanced_query.strip()
            
        except Exception as e:
            print(f"Query enhancement failed: {e}")
            return original_query
    
    def generate_persona_aware_sql(self, query_context: PersonaQueryContext, 
                                   metadata: str, examples: str) -> str:
        """Generate SQL with persona-specific optimizations"""
        
        base_sql_prompt = """
        ### Task
        Generate a SQL query to answer: {enhanced_query}
        
        ### Persona Context
        User Role: {persona_type}
        Key Focus Areas: {preferred_metrics}
        Confidence: {confidence}
        
        ### Instructions
        - Optimize the query for {persona_type} insights
        - Include relevant KPIs: {preferred_metrics}
        - If confidence is low, provide a general analytical response
        - Focus on metrics that matter to this role
        
        ### Database Schema
        {metadata}
        
        ### Examples
        {examples}
        
        ### Answer
        Given the database schema and persona context, here is the SQL query:
        [SQL]
        """
        
        try:
            prompt = PromptTemplate(
                template=base_sql_prompt,
                input_variables=["enhanced_query", "persona_type", "preferred_metrics", 
                               "confidence", "metadata", "examples"]
            )
            
            chain = LLMChain(prompt=prompt, llm=self.llm_model)
            
            sql_response = chain.run(
                enhanced_query=query_context.enhanced_query,
                persona_type=query_context.persona_type.value.replace("_", " ").title(),
                preferred_metrics=", ".join(query_context.preferred_metrics),
                confidence=f"{query_context.confidence:.2f}",
                metadata=metadata,
                examples=examples
            )
            
            return self._extract_sql_from_response(sql_response)
            
        except Exception as e:
            print(f"SQL generation failed: {e}")
            return "SELECT 'Error generating SQL' as message"
    
    def format_persona_response(self, query_context: PersonaQueryContext, 
                               data_results: pd.DataFrame, sql_query: str) -> str:
        """Format response according to persona preferences"""
        
        if query_context.persona_type == PersonaType.UNKNOWN:
            return self._format_standard_response(data_results, sql_query)
        
        response_template = self.persona_templates[query_context.persona_type]["response_format"]
        
        try:
            # Prepare data summary
            data_summary = self._create_data_summary(data_results, query_context.persona_type)
            
            # Create query context summary
            context_summary = {
                "persona": query_context.persona_type.value,
                "confidence": query_context.confidence,
                "focus_metrics": query_context.preferred_metrics,
                "original_query": query_context.original_query
            }
            
            prompt = PromptTemplate(
                template=response_template,
                input_variables=["data_summary", "query_context"]
            )
            
            chain = LLMChain(prompt=prompt, llm=self.llm_model)
            
            formatted_response = chain.run(
                data_summary=data_summary,
                query_context=json.dumps(context_summary, indent=2)
            )
            
            return formatted_response.strip()
            
        except Exception as e:
            print(f"Response formatting failed: {e}")
            return self._format_standard_response(data_results, sql_query)
    
    def _create_data_summary(self, data_results: pd.DataFrame, persona_type: PersonaType) -> str:
        """Create a persona-specific summary of the data"""
        
        if data_results.empty:
            return "No data found for the specified query."
        
        summary_parts = []
        
        # Basic stats
        summary_parts.append(f"Dataset contains {len(data_results)} records")
        
        # Persona-specific insights
        if persona_type == PersonaType.SALES_MANAGER:
            # Focus on revenue and performance metrics
            numeric_cols = data_results.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                revenue_like = [col for col in numeric_cols if 'revenue' in col.lower() or 'total' in col.lower()]
                if revenue_like:
                    for col in revenue_like[:2]:  # Top 2 revenue metrics
                        total = data_results[col].sum()
                        summary_parts.append(f"Total {col}: ${total:,.2f}")
                        
        elif persona_type == PersonaType.PRODUCT_ANALYST:
            # Focus on product and margin metrics
            if 'ProductName' in data_results.columns:
                unique_products = data_results['ProductName'].nunique()
                summary_parts.append(f"Analysis covers {unique_products} unique products")
            
            profit_cols = [col for col in data_results.columns if 'profit' in col.lower() or 'margin' in col.lower()]
            if profit_cols:
                for col in profit_cols[:2]:
                    if data_results[col].dtype in ['int64', 'float64']:
                        avg_value = data_results[col].mean()
                        summary_parts.append(f"Average {col}: ${avg_value:,.2f}")
        
        elif persona_type == PersonaType.CUSTOMER_SUCCESS:
            # Focus on customer metrics
            if 'CustomerID' in data_results.columns or 'FullName' in data_results.columns:
                customer_col = 'CustomerID' if 'CustomerID' in data_results.columns else 'FullName'
                unique_customers = data_results[customer_col].nunique()
                summary_parts.append(f"Analysis covers {unique_customers} customers")
                
        elif persona_type == PersonaType.EXECUTIVE:
            # Focus on high-level metrics
            numeric_cols = data_results.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary_parts.append(f"Key metrics analyzed: {', '.join(numeric_cols[:3])}")
        
        return " | ".join(summary_parts)
    
    def _get_preferred_visualization(self, persona_type: PersonaType) -> str:
        """Get the preferred visualization type for a persona"""
        viz_preferences = {
            PersonaType.SALES_MANAGER: "line_chart",  # For trend analysis
            PersonaType.PRODUCT_ANALYST: "bar_chart",  # For comparative analysis
            PersonaType.CUSTOMER_SUCCESS: "scatter_plot",  # For relationship analysis
            PersonaType.EXECUTIVE: "dashboard",  # For high-level overview
            PersonaType.UNKNOWN: "table"
        }
        return viz_preferences.get(persona_type, "table")
    
    def _extract_sql_from_response(self, llm_response: str) -> str:
        """Extract SQL query from LLM response"""
        # Look for SQL code blocks
        if '```sql' in llm_response:
            start = llm_response.find('```sql') + 6
            end = llm_response.find('```', start)
            if end != -1:
                return llm_response[start:end].strip()
        
        # Look for [SQL] markers
        if '[SQL]' in llm_response:
            start = llm_response.find('[SQL]') + 5
            return llm_response[start:].strip()
        
        # Return as-is if no markers found
        return llm_response.strip()
    
    def _format_standard_response(self, data_results: pd.DataFrame, sql_query: str) -> str:
        """Format a standard response when persona is unknown"""
        if data_results.empty:
            return "No data found for your query."
        
        summary = f"Found {len(data_results)} records. Here's what the data shows:"
        
        # Add basic insights
        numeric_cols = data_results.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # Show top 3 numeric columns
                if data_results[col].notna().any():
                    total = data_results[col].sum()
                    avg = data_results[col].mean()
                    summary += f"\n- {col}: Total = {total:,.2f}, Average = {avg:,.2f}"
        
        return summary
    
    def get_persona_insights(self, query_context: PersonaQueryContext, 
                           data_results: pd.DataFrame) -> Dict[str, Any]:
        """Generate persona-specific insights and recommendations"""
        
        insights = {
            "persona_type": query_context.persona_type.value,
            "confidence": query_context.confidence,
            "key_metrics": [],
            "recommendations": [],
            "quick_actions": self.persona_detector.get_quick_actions(query_context.persona_type),
            "visualization_suggestion": query_context.visualization_type
        }
        
        if data_results.empty:
            insights["recommendations"].append("No data available for analysis. Consider refining your query.")
            return insights
        
        # Generate persona-specific insights
        if query_context.persona_type == PersonaType.SALES_MANAGER:
            insights.update(self._generate_sales_insights(data_results))
        elif query_context.persona_type == PersonaType.PRODUCT_ANALYST:
            insights.update(self._generate_product_insights(data_results))
        elif query_context.persona_type == PersonaType.CUSTOMER_SUCCESS:
            insights.update(self._generate_customer_insights(data_results))
        elif query_context.persona_type == PersonaType.EXECUTIVE:
            insights.update(self._generate_executive_insights(data_results))
        
        return insights
    
    def _generate_sales_insights(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Generate sales-specific insights"""
        insights = {"key_metrics": [], "recommendations": []}
        
        # Revenue analysis
        revenue_cols = [col for col in data.columns if 'revenue' in col.lower() or 'total' in col.lower()]
        if revenue_cols:
            for col in revenue_cols[:2]:
                if data[col].dtype in ['int64', 'float64']:
                    total_revenue = data[col].sum()
                    insights["key_metrics"].append(f"Total {col}: ${total_revenue:,.2f}")
                    
                    # Performance recommendations
                    if data[col].std() > data[col].mean() * 0.5:
                        insights["recommendations"].append("High revenue variance detected - investigate top and bottom performers")
        
        # Territory analysis
        if 'Territory' in data.columns:
            top_territory = data.groupby('Territory')[revenue_cols[0]].sum().idxmax() if revenue_cols else None
            if top_territory:
                insights["recommendations"].append(f"Focus on replicating {top_territory} territory success strategies")
        
        return insights
    
    def _generate_product_insights(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Generate product-specific insights"""
        insights = {"key_metrics": [], "recommendations": []}
        
        # Margin analysis
        margin_cols = [col for col in data.columns if 'margin' in col.lower() or 'profit' in col.lower()]
        if margin_cols:
            for col in margin_cols[:2]:
                if data[col].dtype in ['int64', 'float64']:
                    avg_margin = data[col].mean()
                    insights["key_metrics"].append(f"Average {col}: ${avg_margin:,.2f}")
                    
                    # Margin recommendations
                    low_margin_threshold = data[col].quantile(0.25)
                    insights["recommendations"].append(f"Review products with margins below ${low_margin_threshold:,.2f}")
        
        # Product performance
        if 'ProductName' in data.columns and revenue_cols:
            product_count = data['ProductName'].nunique()
            insights["key_metrics"].append(f"Products analyzed: {product_count}")
            insights["recommendations"].append("Consider product portfolio optimization based on performance data")
        
        return insights
    
    def _generate_customer_insights(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Generate customer-specific insights"""
        insights = {"key_metrics": [], "recommendations": []}
        
        # Customer metrics
        customer_cols = [col for col in data.columns if 'customer' in col.lower()]
        if customer_cols or 'FullName' in data.columns:
            customer_col = customer_cols[0] if customer_cols else 'FullName'
            unique_customers = data[customer_col].nunique()
            insights["key_metrics"].append(f"Customers analyzed: {unique_customers}")
        
        # Lifetime value analysis
        ltv_cols = [col for col in data.columns if 'lifetime' in col.lower() or 'value' in col.lower()]
        if ltv_cols:
            for col in ltv_cols[:1]:
                if data[col].dtype in ['int64', 'float64']:
                    avg_ltv = data[col].mean()
                    insights["key_metrics"].append(f"Average {col}: ${avg_ltv:,.2f}")
                    insights["recommendations"].append("Focus retention efforts on high-value customers")
        
        return insights
    
    def _generate_executive_insights(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Generate executive-level insights"""
        insights = {"key_metrics": [], "recommendations": []}
        
        # High-level metrics
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:
                total_value = data[col].sum()
                insights["key_metrics"].append(f"Total {col}: {total_value:,.2f}")
        
        # Strategic recommendations
        insights["recommendations"].extend([
            "Monitor key performance trends for strategic planning",
            "Consider market expansion opportunities based on current performance",
            "Review resource allocation across high-performing areas"
        ])
        
        return insights

# Integration class to tie everything together
class PersonaAwareBusinessIntelligence:
    """Main class that integrates persona detection with query processing"""
    
    def __init__(self, llm_model, metadata: str, examples: str):
        self.llm_model = llm_model
        self.metadata = metadata
        self.examples = examples
        self.persona_detector = PersonaDetector(llm_model)
        self.query_processor = PersonaQueryProcessor(llm_model, self.persona_detector)
        self.conversation_history = []
    
    def process_business_query(self, user_query: str) -> Dict[str, Any]:
        """Main method to process a business query with persona awareness"""
        
        # Store conversation
        self.conversation_history.append(user_query)
        
        # Process query with persona awareness
        query_context = self.query_processor.process_query(
            user_query, self.conversation_history
        )
        
        # Generate persona-aware SQL
        sql_query = self.query_processor.generate_persona_aware_sql(
            query_context, self.metadata, self.examples
        )
        
        # Execute SQL (you'll need to integrate this with your existing sql_query_execution function)
        # data_results = execute_sql_query(sql_query)  # This would be your existing function
        
        # For now, return the structure that would be used
        return {
            "query_context": query_context,
            "sql_query": sql_query,
            "persona_type": query_context.persona_type.value,
            "confidence": query_context.confidence,
            "enhanced_query": query_context.enhanced_query,
            "preferred_metrics": query_context.preferred_metrics,
            "visualization_type": query_context.visualization_type,
            "quick_actions": self.persona_detector.get_quick_actions(query_context.persona_type)
        }
    
    def format_response_for_persona(self, query_context: PersonaQueryContext, 
                                   data_results: pd.DataFrame, sql_query: str) -> Dict[str, Any]:
        """Format the complete response for the detected persona"""
        
        # Format the main response
        formatted_response = self.query_processor.format_persona_response(
            query_context, data_results, sql_query
        )
        
        # Get persona-specific insights
        insights = self.query_processor.get_persona_insights(
            query_context, data_results
        )
        
        return {
            "formatted_response": formatted_response,
            "insights": insights,
            "persona_context": {
                "type": query_context.persona_type.value,
                "confidence": query_context.confidence,
                "response_style": self.persona_detector.get_response_style(query_context.persona_type)
            },
            "data_summary": data_results.describe().to_dict() if not data_results.empty else {},
            "sql_query": sql_query
        }

# Example usage and testing
if __name__ == "__main__":
    # This would integrate with your existing model setup
    from langchain.chat_models import ChatOpenAI
    
    # Initialize with your existing components
    llm_model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    # Load your existing metadata and examples
    # Load metadata and examples from files
    with open('/Users/jaishankarbhagat/Documents/AI Hub/Agentic-chatbot/Chatbot/metadata_adv.txt', 'r') as metadata_file:
        metadata = metadata_file.read()
    
    with open('/Users/jaishankarbhagat/Documents/AI Hub/Agentic-chatbot/Chatbot/query_example.txt', 'r') as examples_file:
        examples = examples_file.read()
    
    # Create the persona-aware system
    persona_bi = PersonaAwareBusinessIntelligence(llm_model, metadata, examples)
    
    # Test with different persona queries
    test_queries = [
        "What's our monthly sales performance by territory?",  # Sales Manager
        "Which products have the highest profit margins?",     # Product Analyst
        "Show me our top customers and their purchase history", # Customer Success
        "Give me an executive summary of Q4 performance"       # Executive
    ]
    
    for query in test_queries:
        result = persona_bi.process_business_query(query)
        print(f"\nQuery: {query}")
        print(f"Detected Persona: {result['persona_type']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Enhanced Query: {result['enhanced_query']}")
        print(f"SQL Query: {result['sql_query']}")
        print("=" * 80)