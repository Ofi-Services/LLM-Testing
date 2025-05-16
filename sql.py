from langchain_ollama import OllamaLLM
from typing_extensions import TypedDict
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from db_create import CargaDeArchivos
import re
import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import login

def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def count_tokens(text: str,tokenizer) -> int:
    """
    Count the number of tokens in a given text using the Mistral tokenizer."
    """
    # Tokenize the text and return the number of tokens
    return len(tokenizer.encode(text))

class State(TypedDict):
    """
    Represents the state of the workflow, including the question, schema, database connection,
    relevance, SQL query, query result, and other metadata.
    """
    question: str
    db_conn: None
    query_df: pd.DataFrame
    sql_query: str
    query_result: str
    sql_error: bool
    final_answer: str
    attempts: int
    prompt: str
    tokenizer: None

def identify_question_type(q: str) -> str:
    q = q.lower()
    if any(w in q for w in ["average", "mean", "duration", "time taken", "how long"]):
        return "average"
    if any(w in q for w in ["distribution", "frequency", "histogram"]):
        return "distribution"
    if any(w in q for w in ["trend", "over time", "change", "evolution"]):
        return "trend"
    if any(w in q for w in ["most", "top", "highest", "least", "lowest", "compare"]):
        return "ranking"
    return "general"
    
def summarize_dataframe(df: pd.DataFrame, question_type: str) -> str:
    summary = ""

    if df.empty:
        return "⚠️ No data to summarize."

    if question_type == "average":
        numeric_cols = df.select_dtypes(include="number")
        if not numeric_cols.empty:
            summary += numeric_cols.mean().to_frame("mean").T.to_string()
        else:
            summary += "ℹ️ No numeric columns to compute averages."
    elif question_type == "distribution":
        for col in df.select_dtypes(include=["object", "category"]):
            dist = df[col].value_counts(normalize=True).head(3)
            summary += f"\n- {col}: {dist.to_dict()}"
    elif question_type == "trend":
        time_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if time_cols:
            col = time_cols[0]
            df_sorted = df.sort_values(by=col)
            summary += f"Sample over time ({col}):\n"
            summary += df_sorted[[col]].head(5).to_string(index=False)
        else:
            summary += "ℹ️ No time-related column found to show trend."
    elif question_type == "ranking":
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) >= 1:
            col = numeric_cols[0]
            top = df.nlargest(3, col)[[col]].to_string(index=False)
            summary += f"Top 3 rows by {col}:\n{top}"
        else:
            summary += "ℹ️ No numeric column found for ranking."
    else:  # General fallback
        summary += df.describe(include='all').to_string()
    return summary

def convert_nl_to_sql(state: State):
    """
    Converts a natural language question into an SQL query based on the database schema.
 
    Args:
        state (State): The current state of the workflow.
 
    Returns:
        State: Updated state with the generated SQL query.
    """
    question = state["question"]
    # Seleccionar el prompt apropiado basado en el caso de uso
    system = state["prompt"] 
    # Agregar información específica sobre case sensitivity y estructura de tablas
    # Añadir las notas adicionales al prompt del sistema
    llm = OllamaLLM(model="qwen3:4b", temperature="0.0")
 
    convert_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Question: {question}"),
            ]
        )
    sql_generator = convert_prompt | llm
    
    print(f"Converting question to SQL {question}")
    result = sql_generator.invoke({"question": question})
    result= remove_think_blocks(result)
    # Limpiar el código SQL eliminando los marcadores de bloque de código
    message = re.sub(r'^\s*```sql\s*|\s*```$', '', result.strip(), flags=re.IGNORECASE)
    # Corrección adicional para asegurar capitalización correcta de valores de confianza
    message = re.sub(r"confidence\s*=\s*'high'", "confidence = 'High'", message, flags=re.IGNORECASE)
    message = re.sub(r"confidence\s*=\s*'medium'", "confidence = 'Medium'", message, flags=re.IGNORECASE)
    message = re.sub(r"confidence\s*=\s*'low'", "confidence = 'Low'", message, flags=re.IGNORECASE)
    # Corrección para el acceso a campo supplier en grouped.items
    # Solo si la consulta está usando la tabla grouped y tratando de acceder directamente a supplier
    if "grouped" in message and "supplier" in message and "item.case.supplier" not in message:
        message = re.sub(r"([^.])supplier", r"\1item.case.supplier", message)
    print(f"Generated SQL query: {message}")
    state["sql_query"] = message  # Store the generated SQL query in the state
    state["attempts"] = 0 # Initialize attempts to 0
    return state



def execute_sql(state:State):
    """
    Executes the SQL query on the  database and retrieves the results.

    Args:
        state (State): The current state of the workflow.
        config (RunnableConfig): Configuration for the runnable.

    Returns:
        State: Updated state with the query results or error information.
    """
    
    # If multiple queries are generated, execute them one by one
    db_conn = state["db_conn"] 
    query = state["sql_query"]
    error = state.get("sql_error", True)  # Default: all True (assume they need execution)
    result = state.get("query_result", None)
    dataframe = state.get("query_df", None)
    if error or result is None:  # Execute if error OR never executed before
        print(f"🚀 Executing query: {query}")
        try:
            # Ensure the query targets only the allowed tables
            allowed_tables = ["cases", "activities","variants","grouped","invoices"]
            if not any(table in query.lower() for table in allowed_tables):
                raise ValueError(f"Query must target only the tables: {', '.join(allowed_tables)}.")

            # Execute the SQL query using the connection
            cursor = db_conn.cursor()
            cursor.execute(query)

            # Fetch results if it's a SELECT query
            if query.lower().startswith("select"):
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                # Format the output
                if rows:
                    formatted_result = "\n".join(
                        ", ".join(f"{col}: {row[idx]}" for idx, col in enumerate(columns))
                        for row in rows
                    )
                    print("SQL SELECT query executed successfully.")
                    
                else:
                    formatted_result = "No results found."
                    print("SQL SELECT query executed successfully but returned no rows.")

                df = pd.DataFrame(rows, columns=columns)
                dataframe = df  # Store the DataFrame in the state
            else:
                formatted_result = "The action has been successfully completed."
                print("SQL command executed successfully.")

            result= formatted_result
            error= False # Mark this query as executed successfully

        except Exception as e:
            result=f"Error executing SQL query: {str(e)}" # Store the error message in the results
            error= True # Mark this query as executed with an error
            print(f"Error executing SQL query: {str(e)}")
    state["query_result"] = result  # Store the list of query results in the state
    state["sql_error"] = error  # Store the list of error states in the state
    state["query_df"] = dataframe  # Store the list of DataFrames in the state
    print(f"SQL query results: {state['query_result']}")
    print(f"SQL error states: {state['sql_error']}")
    return state



def generate_serious_answer(state: State):
    """
    Generates a business-oriented response using SQL query results from sub-questions
    to answer the main question.
    
    Args:
        state (State): The current state of the workflow.
        
    Returns:
        State: Updated state with the final answer.
    """
    question = state["question"]
    query_result = state["query_result"]  # This is now a list of results, one per sub-question

    # Concatenate each sub-question with its answer
    system = f"""
    /no_think
    You are ✨SOFIA✨, an AI business assistant. 
    Your task is to answer the user's question based on the SQL query results.
    Use the SQL query results to support your answer.
    If the SQL query results are empty, indicate you were not able to processe the question.

    SQL query results:
    {query_result}
    """


    human_message = f"Question: {question}"
    
    # Use sOFIa to generate a response based on the SQL result
    llm = OllamaLLM(model="qwen3:4b", temperature="0.0", max_tokens=200)
    response = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human_message),
    ]) | llm | StrOutputParser()
    
    # Generate and store the response
    message = response.invoke({})
    message = remove_think_blocks(message)
    state["final_answer"] = message
    return state



def regenerate_query(state):
    """
    Fixes the SQL query by passing the error message to the SQL model instead of rewriting the user's question.

    Args:
        state (State): The current state of the workflow.

    Returns:
        State: Updated state with the fixed query.
    """

    error = state["query_result"]
    query = state["sql_query"]

    llm = OllamaLLM(model="qwen3:4b", temperature=0.0)
        
    print(f"⚠️ Fixing SQL query: {query}")
    print(f"🔍 Error encountered: {error}")
    part1= f"""
            /no_think
            You are an expert in SQL for DuckDB.
            Your task is to correct the following SQL query based on the error message.

            ### **Query to Fix:**
            ```sql
            {query}
            ```

            ### **Error Message:**
            {error}

            Provide a **corrected** SQL query that runs successfully in the following database schema.
            """
    part_2= state["prompt"]
    sql_fix_prompt = ChatPromptTemplate.from_messages([(
            "system", 
            part1+part_2),
            ("human", "Fix the query and return only the corrected SQL, no explanations."),
        ])

    fixer = sql_fix_prompt | llm 
    # Pass the query and error message to the SQL model for correction
    corrected_query = fixer.invoke({"query": query, "error": error})
    corrected_query = remove_think_blocks(corrected_query)
    # Extract only the SQL code from a markdown block like ```sql ... ``` 
    corrected_query = re.sub(r"```sql\s*(.*?)\s*```", r"\1", corrected_query.strip(), flags=re.DOTALL | re.IGNORECASE)

    state["sql_query"] = corrected_query
    print(f"✅ Fixed SQL query: {corrected_query}")
    state["attempts"] += 1
    return state



def summarize_results(state: dict) -> dict:
    """
    Summarizes query results with more than 1000 tokens.
    The summary is based on the context of the related question or falls back to general statistics.

    Args:
        state (dict): Workflow state containing questions, dataframes, and results.

    Returns:
        dict: Updated state with summarized query results.
    """
    result = state["query_result"]
    dataframe = state["query_df"]
    question = state["question"]
    tokenizer= state["tokenizer"]
    count= count_tokens(result,tokenizer)
    # Check if the result is a list of dataframes and if any of them exceed 2000 tokens
    print(f"Token count: {count}")
    if count <= 4000:

        return state  # No need to summarize if the result is already concise

    df = dataframe
    question = question
    question_type = identify_question_type(question)

    summary = f"📊 Summary of result:\n"
    summary += f"- Rows: {len(df)}\n"
    summary += f"- Columns: {', '.join(df.columns)}\n\n"
    summary += f"🔹 Type: {question_type.capitalize()}-based Summary:\n"
    summary += summarize_dataframe(df, question_type)

    state["query_result"] = summary
    print(f"✅ Summarized result ({question_type} type, >1000 tokens)")
    return state


def end_max_iterations(state: State):
    """
    Ends the workflow after reaching the maximum number of attempts.

    Args:
        state (State): The current state of the workflow.
        config (RunnableConfig): Configuration for the runnable.

    Returns:
        State: Updated state with a termination message.
    """
    state["query_result"] = "Please try again."
    state["final_answer"] = "I couldn't generate a valid SQL query after 3 attempts. Please try again."
    print("Maximum attempts reached. Ending the workflow.")
    return state

def check_attempts_router(state: State):
    """
    Routes the workflow based on the number of attempts made to generate a valid SQL query.

    Args:
        state (State): The current state of the workflow.

    Returns:
        str: The next node in the workflow.
    """
    if state["attempts"] <= 3:
        print(f"Attempt {state['attempts']}")
        return "Retries < 3"
    else:
        return "Retries >= 3"



def execute_sql_router(state: State):
    """
    Routes the workflow based on whether the SQL query execution was successful.

    Args:
        state (State): The current state of the workflow.

    Returns:
        str: The next node in the workflow.
    """
    error= state["sql_error"]
    if error == True:
        return "Error"
    else:
        return "Success"

def run_sql_workflow(question, prompt):
    """
    Run the SQL workflow with the given parameters.

    Args:
        question (str): The natural language question to be converted to SQL.
        prompt (str): The prompt given by the tool call.

    Returns:
        str: The final answer generated by the workflow.
    """
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    a= CargaDeArchivos()
    a.run()
    db_conn= a.conn
    login(token="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")
    workflow = StateGraph(State)
    workflow.add_node("Generates SQL queries", convert_nl_to_sql)
    workflow.add_node("Executes SQL",execute_sql)
    workflow.add_node("Regenerate Error-Queries",regenerate_query)
    workflow.add_node("Answer Relevant Question",generate_serious_answer)
    workflow.add_node("Stops due to max Iterations",end_max_iterations)
    workflow.add_node("Summarizes Results", summarize_results)

    workflow.add_edge(START, "Generates SQL queries")


    workflow.add_edge("Generates SQL queries", "Executes SQL")


    workflow.add_conditional_edges(
            "Executes SQL",
            execute_sql_router,
            {
                "Success": "Summarizes Results",
                "Error": "Regenerate Error-Queries",
            },
        )

    workflow.add_edge("Summarizes Results", "Answer Relevant Question")

    workflow.add_conditional_edges(
            "Regenerate Error-Queries",
            check_attempts_router,
            {
                "Retries < 3": "Executes SQL",
                "Retries >= 3": "Stops due to max Iterations",
            },
        )
    workflow.add_edge("Stops due to max Iterations", END)
    workflow.add_edge("Answer Relevant Question",END)
    chain= workflow.compile()
    state = chain.invoke({"question": question, "db_conn": db_conn,"prompt":prompt,"tokenizer":tokenizer})
    return state["final_answer"]