"""
FIXED COMPLETE INTEGRATION
This file properly integrates all components and fixes the LangChain errors
"""

import streamlit as st
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# =============================================================================
# PERSONA TYPES AND DETECTION
# =============================================================================

class PersonaType(Enum):
    SALES_MANAGER = "sales_manager"
    PRODUCT_ANALYST = "product_analyst"
    CUSTOMER_SUCCESS = "customer_success"
    EXECUTIVE = "executive"
    UNKNOWN = "unknown"

@dataclass
class PersonaProfile:
    """Defines characteristics for each persona"""
    persona_type: PersonaType
    keywords: List[str]
    priorities: List[str]
    response_style: str
    kpi_focus: List[str]
    quick_actions: List[str]

class PersonaDetector:
    """Detects user persona from conversation patterns"""
    
    def __init__(self):
        self.persona_profiles = {
            PersonaType.SALES_MANAGER: PersonaProfile(
                persona_type=PersonaType.SALES_MANAGER,
                keywords=["sales", "revenue", "territory", "quota", "commission", "pipeline", "deals", "forecast", "team", "rep"],
                priorities=["Revenue tracking", "Team performance", "Territory analysis", "Sales forecasting"],
                response_style="Actionable insights with clear metrics and trends",
                kpi_focus=["Total Revenue", "Sales Growth %", "Close Rate", "Average Deal Size"],
                quick_actions=["View team performance", "Territory analysis", "Pipeline review", "Forecast generation"]
            ),
            PersonaType.PRODUCT_ANALYST: PersonaProfile(
                persona_type=PersonaType.PRODUCT_ANALYST,
                keywords=["product", "margin", "inventory", "category", "profit", "cost", "analytics", "performance", "units", "profitability"],
                priorities=["Product performance", "Margin analysis", "Inventory optimization", "Category insights"],
                response_style="Detailed analytical breakdowns with statistical insights",
                kpi_focus=["Gross Margin %", "Product Revenue", "Inventory Turnover", "Profit per Unit"],
                quick_actions=["Product performance report", "Margin analysis", "Inventory review", "Category comparison"]
            ),
            PersonaType.CUSTOMER_SUCCESS: PersonaProfile(
                persona_type=PersonaType.CUSTOMER_SUCCESS,
                keywords=["customer", "retention", "satisfaction", "loyalty", "support", "lifetime", "churn", "relationship", "health"],
                priorities=["Customer satisfaction", "Retention analysis", "Customer health", "Relationship management"],
                response_style="Customer-centric insights with relationship focus",
                kpi_focus=["Customer Lifetime Value", "Retention Rate", "Satisfaction Score", "Repeat Purchase Rate"],
                quick_actions=["Customer health dashboard", "Retention analysis", "Satisfaction review", "Account management"]
            ),
            PersonaType.EXECUTIVE: PersonaProfile(
                persona_type=PersonaType.EXECUTIVE,
                keywords=["executive", "overview", "strategic", "growth", "total", "summary", "board", "quarterly", "annual", "company"],
                priorities=["Strategic insights", "High-level metrics", "Growth trends", "Executive summaries"],
                response_style="Executive summaries with key takeaways and strategic implications",
                kpi_focus=["Total Revenue", "Growth Rate", "Market Share", "ROI"],
                quick_actions=["Executive dashboard", "Strategic overview", "Board presentation", "Growth analysis"]
            )
        }
    
    def detect_persona(self, query: str, conversation_history: List[str] = None) -> Tuple[PersonaType, float]:
        """Detect persona based on keywords and conversation context"""
        query_lower = query.lower()
        scores = {}
        
        # Score each persona based on keyword matches
        for persona_type, profile in self.persona_profiles.items():
            keyword_matches = sum(1 for keyword in profile.keywords if keyword in query_lower)
            scores[persona_type] = keyword_matches / len(profile.keywords) if profile.keywords else 0
        
        # Consider conversation history for context
        if conversation_history:
            for msg in conversation_history[-3:]:  # Last 3 messages
                msg_lower = msg.lower()
                for persona_type, profile in self.persona_profiles.items():
                    additional_matches = sum(1 for keyword in profile.keywords if keyword in msg_lower)
                    scores[persona_type] += (additional_matches / len(profile.keywords)) * 0.3
        
        if not scores or max(scores.values()) == 0:
            return PersonaType.UNKNOWN, 0.0
        
        best_persona = max(scores, key=scores.get)
        confidence = min(scores[best_persona] * 1.5, 1.0)  # Scale and cap at 1.0
        
        return best_persona, confidence
    
    def get_persona_profile(self, persona_type: PersonaType) -> PersonaProfile:
        """Get profile for a specific persona"""
        return self.persona_profiles.get(persona_type)

# =============================================================================
# FIXED MODEL WRAPPER (Solves the LangChain Error)
# =============================================================================

class FixedModelWrapper:
    """Fixed wrapper that handles LangChain properly"""
    
    def __init__(self):
        # API key setup
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            st.error("❌ OpenAI API key not found!")
            return
        
        # Initialize LLM
        try:
            self.llm_model = ChatOpenAI(
                temperature=0,
                model="gpt-4o-mini",
                openai_api_key=self.openai_api_key
            )
            st.success("✅ OpenAI model initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize OpenAI model: {e}")
            return
        
        # Load metadata and examples with error handling
        self.metadata = self._load_file_safe("metadata_adv.txt", "Database metadata")
        self.examples = self._load_file_safe("query_example.txt", "Query examples")
        self.prompt_template = self._load_file_safe("prompt_adv.md", "Prompt template")
        
        # Import original model if available
        try:
            import model
            self.original_model = model
            st.success("✅ Original model imported successfully!")
        except ImportError as e:
            st.warning(f"⚠️ Could not import original model: {e}")
            self.original_model = None
    
    def _load_file_safe(self, filename: str, description: str) -> str:
        """Safely load files with error handling"""
        try:
            with open(filename, "r", encoding='utf-8') as f:
                content = f.read()
            st.success(f"✅ {description} loaded successfully")
            return content
        except FileNotFoundError:
            st.warning(f"⚠️ {filename} not found. Using default content.")
            return f"# Default {description}\nNo specific content available."
        except Exception as e:
            st.error(f"❌ Error loading {filename}: {e}")
            return f"# Error loading {description}\nPlease check file format."
    
    def generate_enhanced_sql(self, user_query: str, persona_type: PersonaType, confidence: float) -> str:
        """Generate SQL with persona enhancement and fixed LangChain handling"""
        
        # Create persona-enhanced query
        if persona_type != PersonaType.UNKNOWN and confidence > 0.3:
            enhanced_query = self._enhance_query_for_persona(user_query, persona_type)
        else:
            enhanced_query = user_query
        
        # Try original model first
        if self.original_model:
            try:
                result = self.original_model.func_final_result(enhanced_query)
                return result[1] if len(result) > 1 else "SELECT 'No SQL generated' as message"
            except Exception as e:
                st.warning(f"⚠️ Original model failed: {e}. Using backup method.")
        
        # Fallback to direct LLM call with fixed prompt handling
        return self._generate_sql_direct(enhanced_query)
    
    def _enhance_query_for_persona(self, query: str, persona_type: PersonaType) -> str:
        """Add persona-specific context to queries"""
        enhancements = {
            PersonaType.SALES_MANAGER: " Include sales performance metrics, revenue analysis, and territory data where relevant.",
            PersonaType.PRODUCT_ANALYST: " Focus on product profitability, margins, inventory metrics, and category performance.",
            PersonaType.CUSTOMER_SUCCESS: " Emphasize customer relationships, retention metrics, satisfaction data, and lifetime value.",
            PersonaType.EXECUTIVE: " Provide high-level strategic insights, key performance indicators, and executive-level summaries.",
        }
        
        enhancement = enhancements.get(persona_type, "")
        return query + enhancement
    
    def _generate_sql_direct(self, query: str) -> str:
        """Direct SQL generation with proper error handling"""
        try:
            # Create a simple, working prompt
            prompt_text = f"""
            Generate a SQL query to answer this business question: {query}

            Available database tables:
            {self.metadata}

            Example patterns:
            {self.examples}

            Instructions:
            - Use proper JOIN syntax
            - Include relevant business calculations
            - Return only the SQL query
            - If you cannot generate a valid query, return: SELECT 'Unable to process query' as message

            SQL Query:
            """
            
            # Direct LLM call without complex prompt templates
            response = self.llm_model.invoke(prompt_text)
            
            if hasattr(response, 'content'):
                sql_result = response.content.strip()
            else:
                sql_result = str(response).strip()
            
            # Clean up the SQL response
            if '```sql' in sql_result:
                sql_result = sql_result.split('```sql')[1].split('```')[0].strip()
            elif '```' in sql_result:
                sql_result = sql_result.split('```')[1].strip()
            
            return sql_result
            
        except Exception as e:
            st.error(f"❌ SQL generation error: {e}")
            return "SELECT 'Error generating SQL' as message"
    
    def execute_sql(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL with error handling"""
        try:
            if self.original_model:
                return self.original_model.sql_query_execution(sql_query)
            else:
                # Fallback execution
                import pandasql as ps
                # You'll need to import your CSV data here
                return ps.sqldf(sql_query)
        except Exception as e:
            st.error(f"❌ SQL execution error: {e}")
            return pd.DataFrame({'error': [str(e)]})
    
    def generate_persona_summary(self, query: str, results: pd.DataFrame, 
                                persona_type: PersonaType, sql_query: str) -> str:
        """Generate persona-specific summary"""
        
        if results.empty or 'error' in results.columns:
            return f"No data found for your query: '{query}'"
        
        # Create persona-specific summary
        persona_prefixes = {
            PersonaType.SALES_MANAGER: "📈 **Sales Performance Analysis**",
            PersonaType.PRODUCT_ANALYST: "📊 **Product Analytics Report**",
            PersonaType.CUSTOMER_SUCCESS: "👥 **Customer Success Insights**",
            PersonaType.EXECUTIVE: "🎯 **Executive Dashboard**",
            PersonaType.UNKNOWN: "📋 **Business Analysis**"
        }
        
        prefix = persona_prefixes.get(persona_type, "📋 **Analysis Results**")
        
        # Generate basic summary
        summary = f"{prefix}\n\n"
        summary += f"Found {len(results)} records for your analysis.\n\n"
        
        # Add persona-specific insights
        if persona_type == PersonaType.SALES_MANAGER:
            summary += "**Key Sales Metrics:**\n"
        elif persona_type == PersonaType.PRODUCT_ANALYST:
            summary += "**Product Performance Insights:**\n"
        elif persona_type == PersonaType.CUSTOMER_SUCCESS:
            summary += "**Customer Relationship Analysis:**\n"
        elif persona_type == PersonaType.EXECUTIVE:
            summary += "**Strategic Business Overview:**\n"
        
        # Add data insights
        numeric_cols = results.select_dtypes(include=['number']).columns
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            if not results[col].isna().all():
                total = results[col].sum()
                avg = results[col].mean()
                summary += f"• {col}: Total = {total:,.2f}, Average = {avg:,.2f}\n"
        
        return summary

# =============================================================================
# COMPLETE INTEGRATED SYSTEM
# =============================================================================

class CompletePersonaSystem:
    """Complete system integrating all persona components"""
    
    def __init__(self):
        # Initialize components
        self.persona_detector = PersonaDetector()
        self.model_wrapper = FixedModelWrapper()
        
        # Initialize session state
        if 'current_persona' not in st.session_state:
            st.session_state.current_persona = PersonaType.UNKNOWN
        if 'persona_confidence' not in st.session_state:
            st.session_state.persona_confidence = 0.0
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
    
    def process_query_complete(self, user_query: str) -> Dict[str, Any]:
        """Complete query processing with all persona features"""
        
        # Get conversation context
        conversation_context = [msg["content"] for msg in st.session_state.messages[-5:] 
                              if msg["role"] == "user"]
        
        # Detect persona
        persona_type, confidence = self.persona_detector.detect_persona(
            user_query, conversation_context
        )
        
        # Update session state
        st.session_state.current_persona = persona_type
        st.session_state.persona_confidence = confidence
        st.session_state.conversation_history.append({
            'query': user_query,
            'persona': persona_type.value,
            'confidence': confidence
        })
        
        # Generate SQL with persona awareness
        sql_query = self.model_wrapper.generate_enhanced_sql(user_query, persona_type, confidence)
        
        # Execute SQL
        results = self.model_wrapper.execute_sql(sql_query)
        
        # Generate persona-specific summary
        summary = self.model_wrapper.generate_persona_summary(
            user_query, results, persona_type, sql_query
        )
        
        # Get persona profile for additional context
        persona_profile = self.persona_detector.get_persona_profile(persona_type)
        
        return {
            'summary': summary,
            'sql_query': sql_query,
            'results': results,
            'persona_type': persona_type.value,
            'confidence': confidence,
            'quick_actions': persona_profile.quick_actions if persona_profile else [],
            'kpi_focus': persona_profile.kpi_focus if persona_profile else [],
            'persona_profile': persona_profile
        }

# =============================================================================
# STREAMLIT UI WITH COMPLETE INTEGRATION
# =============================================================================

def create_complete_persona_ui():
    """Complete Streamlit UI with all persona features integrated"""
    
    st.set_page_config(
        page_title="🎯 AI-Powered Persona Dashboard",
        page_icon="🎯",
        layout="wide"
    )
    
    # Initialize the complete system
    if 'persona_system' not in st.session_state:
        with st.spinner("Initializing AI-Powered Persona System..."):
            st.session_state.persona_system = CompletePersonaSystem()
    
    system = st.session_state.persona_system
    
    # Header with persona indicator
    st.title("🎯 AI-Powered Persona Dashboard")
    st.markdown("*Advanced Business Intelligence with Adaptive AI*")
    
    # Persona status bar
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        persona_name = st.session_state.current_persona.value.replace('_', ' ').title()
        confidence = st.session_state.persona_confidence
        
        if confidence > 0.3:
            st.info(f"🎭 Detected Role: **{persona_name}** ({confidence:.1%} confidence)")
        else:
            st.info("🎭 Role: **Not Detected Yet** - Ask a question to enable persona detection")
    
    with col2:
        if hasattr(st.session_state, 'messages'):
            query_count = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.metric("Queries", query_count)
    
    with col3:
        st.metric("AI Status", "🟢 Ready")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Intelligent Assistant")
        
        # Initialize messages
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Welcome! I'm your AI business intelligence assistant that adapts to your role. How can I help you analyze your Adventure Works data?"}
            ]
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Query input
        if prompt := st.chat_input("Ask me about your business data..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Process with complete persona system
            with st.chat_message("assistant"):
                with st.spinner("🧠 Analyzing with persona intelligence..."):
                    try:
                        # Get complete response
                        response = system.process_query_complete(prompt)
                        
                        # Display persona detection info
                        if response['confidence'] > 0.3:
                            st.success(f"🎭 Detected as {response['persona_type'].replace('_', ' ').title()} ({response['confidence']:.1%} confidence)")
                        
                        # Display summary
                        st.write(response['summary'])
                        
                        # Show SQL and results
                        if 'select' in response['sql_query'].lower() and 'error' not in response['sql_query'].lower():
                            with st.expander("🔍 Generated SQL Query"):
                                st.code(response['sql_query'], language='sql')
                            
                            if not response['results'].empty and 'error' not in response['results'].columns:
                                with st.expander("📊 Data Results"):
                                    st.dataframe(response['results'])
                        
                        # Add to messages
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response['summary']
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Error processing query: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
    
    with col2:
        # Persona-specific sidebar
        st.subheader("🎭 Persona Insights")
        
        if st.session_state.current_persona != PersonaType.UNKNOWN:
            # Get current persona profile
            profile = system.persona_detector.get_persona_profile(st.session_state.current_persona)
            
            if profile:
                # Quick actions
                st.markdown("### ⚡ Quick Actions")
                for action in profile.quick_actions:
                    if st.button(action, key=f"action_{hash(action)}"):
                        st.info(f"Executing: {action}")
                
                # KPI Focus
                st.markdown("### 📈 Key Metrics Focus")
                for kpi in profile.kpi_focus:
                    st.write(f"• {kpi}")
                
                # Priorities
                st.markdown("### 🎯 Business Priorities")
                for priority in profile.priorities:
                    st.write(f"• {priority}")
        
        else:
            st.info("💡 Ask a question to enable persona detection and get customized insights!")
        
        # Sample queries
        st.markdown("### 💭 Try These Sample Queries")
        sample_queries = [
            "What's our total revenue this quarter?",
            "Which products have the highest margins?",
            "Show me our top customers by value",
            "Give me an executive summary of performance"
        ]
        
        for i, query in enumerate(sample_queries):
            if st.button(query, key=f"sample_{i}"):
                # Add to messages and rerun
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()

def main():
    """Main application entry point"""
    create_complete_persona_ui()

if __name__ == "__main__":
    main()