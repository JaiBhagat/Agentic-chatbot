import streamlit as st
import model
import pandas as pd

#Adding page title
st.set_page_config(page_title="Adventure Works DB Chatbot")

with st.sidebar:
    st.title('Adventure Works DB Chatbot')
    st.write('If you want to query the Adventure Works Database, you can ask the bot in simple english. Bot will also respond the answer in simple english along with SQL query and tabular data')
    st.write("\n\n Few points to note.\nIf you want to enquire about a product, please mention the word 'Product' before product name. Like: Product 'Mountain-100 Black, 42'")
    st.write("\n Similarly if you want to enquire about an employee, please mention the word 'Employee' before employee name. For vendor, please use the word 'Vendor', and for customer, 'Customer'.")
    
   
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Adventure Works Chat bot. How may I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            bot_response = model.func_final_result(prompt)
            
            summarized_output = str(bot_response[0])
            sql_query = str(bot_response[1])
            output = model.sql_query_execution(sql_query)
            
            if sql_query.find('select')>=0 and sql_query.find('from')> 7:
                                
                st.write('Thanks for your question. We are happy to help you.')
                if output.shape[0]>0:       
                    for chunks in summarized_output.split("\\n"):
                        st.write(chunks)
                    
                    st.write('Generated SQL Query: \n'+sql_query)
                    st.write('Tabular Output:')
                    st.write(output)    
                else:
                    st.write('No record found')
                    st.write('Generated SQL Query: \n'+sql_query)        
            else:
                st.write('Sorry, I am not sure about the answer you are looking for')    
    message = {"role": "assistant", "content": summarized_output}
    st.session_state.messages.append(message)
