### Task
Generate a SQL query to answer [QUESTION]{user_question}
Addition information for task:
If someone wants to know about any products, we will show product information along with total sales or revenue amount and profit from the product, as mentioned in the example Q4
If someone wants to know about any customers, we will show customer information along with total sales amount for last 5 transactions based on orderdate, as mentioned in the example Q5
If someone wants to know about any employee, we will show employee's information along with total sales amount generated by the employee, as mentioned in the example Q6

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'

### Database Schema
The query will run on a database with the following schema:
{table_metadata_string}

### Example of questions and respective SQL queries
{query_example}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{user_question}
[SQL]
