from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
import os
import pandas as pd
import pandasql as ps
import numpy as np
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain import LLMChain

# Load environment variables from the .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

customers = pd.read_csv("C:\\Gen AI Use Case\\Adventure Works Data\\customers.csv", encoding='ISO-8859-1')
products = pd.read_csv("C:\\Gen AI Use Case\\Adventure Works Data\\products.csv")
productsubcategory = pd.read_csv("C:\\Gen AI Use Case\\Adventure Works Data\\productsubcategories.csv")
productcategory = pd.read_csv("C:\\Gen AI Use Case\\Adventure Works Data\\productcategories.csv")
vendor = pd.read_csv("C:\\Gen AI Use Case\\Adventure Works Data\\vendors.csv")
productvendor = pd.read_csv("C:\\Gen AI Use Case\\Adventure Works Data\\vendorproduct.csv")
employee = pd.read_csv("C:\\Gen AI Use Case\\Adventure Works Data\\employees.csv", encoding='ISO-8859-1')
sales = pd.read_csv("C:\\Gen AI Use Case\\Adventure Works Data\\sales.csv")

#PromptTemplate
def generate_prompt_inference(question, prompt_file="prompt_adv.md", query_example = "query_example.txt", metadata_file="metadata_adv.txt"):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    
    with open(metadata_file, "r") as f:
        table_metadata_string = f.read()
            
    with open(query_example, "r") as f:
        query_example_string = f.read()         

    prompt = prompt.format(
        user_question=question, table_metadata_string=table_metadata_string, query_example=query_example_string
    )
    #prompt = prompt.format(
    #    user_question=question, table_metadata_string=table_metadata_string
    #)
    return prompt
    

#Model Calling
sql_llm_model = ChatOpenAI(
                           temperature=0, model="gpt-4o-mini", openai_api_key=openai_api_key, streaming=True
                          )


def run_inference(question, prompt_file="prompt_adv.md", query_example = "query_example.txt", metadata_file="metadata_adv.txt"):
    prompt = PromptTemplate(
                input_variables = ["question"],
                template = generate_prompt_inference(question)
             )
    
    hub_chain = LLMChain(prompt=prompt, llm=sql_llm_model, verbose=True)
    
    generated_query = hub_chain.run({'inputs':question})
    #generated_query = model(prompt, num_return_sequences=1)
    #llm = HuggingFacePipeline(pipeline = pipe, model_kwargs={"temperature": 0, "max_length": 128, "max_new_tokens": 512},)
    #generated_query = llm(prompt)
    return generated_query

summary_prompt_template = """
You have been provided: 
a. The metadata of database {metadata}, 
b. User question: {question}
c. An sql code to generate a table {sql_code} and 
d. the table output
{table}

You have been asked to convert this table output to simple human language?

"""    
    
def run_summary_inference(output, sql_code, user_question, metadata_file="metadata_adv.txt"):
    prompt = PromptTemplate(
                template = summary_prompt_template.format(metadata=metadata_file, question=user_question, sql_code = sql_code, table=output)
             )
    
    hub_chain = LLMChain(prompt=prompt, llm=sql_llm_model, verbose=True)
    
    generated_summary = hub_chain.run({'inputs':'Convert the given table output to simple human language'})
    
    return generated_summary
    
def sql_query_execution(sql_query):
    try:
        output = ps.sqldf(sql_query)
    except:
        output = pd.DataFrame()    
    return output    
    
def func_final_result(query):
    llm_output = run_inference(query)
    if llm_output.find('```sql\n')>=0:
        query_start_position = llm_output.find('```sql\n')+7
        if llm_output.find('\n```')>0:
            query_end_position = llm_output.find('\n```')
            sql_query = llm_output[query_start_position:query_end_position+1]
        else:
            sql_query = llm_output[query_start_position:]
        
        output = sql_query_execution(sql_query)
    else:
        sql_query = llm_output
        output = pd.DataFrame()
        
    summarized_output = ''
    if output.shape[0]>0:
        summarized_output = run_summary_inference(output, sql_query, query)
    
    final_output = [summarized_output, sql_query]

    return final_output    