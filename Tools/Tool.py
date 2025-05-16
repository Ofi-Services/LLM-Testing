from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
import pandas as pd
import re

# ANALYSIS TOOL
def run_think_task(task: str, context: str = "", use_case: str = "") -> str:
    llm = OllamaLLM(model="mistral:latest", temperature=0.0)
    system_prompt = """
    You are a reasoning engine. Your job is to logically analyze a task, optionally using provided context,
    and generate a clear, accurate response. Be concise, factual, and business-relevant.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context: {context}\nTask: {task}"),
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"task": task, "context": context})
    return result.strip()

# SQL TOOL
class State(TypedDict):
    question: str
    db_conn: any
    query_df: pd.DataFrame
    sql_query: str
    query_result: str
    sql_error: bool
    final_answer: str
    attempts: int
    tokenizer: any
    system_prompt: str
    repair_prompt: str

from transformers import AutoTokenizer

def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))

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
        summary += numeric_cols.mean().to_frame("mean").T.to_string() if not numeric_cols.empty else "ℹ️ No numeric columns to compute averages."
    elif question_type == "distribution":
        for col in df.select_dtypes(include=["object", "category"]):
            dist = df[col].value_counts(normalize=True).head(3)
            summary += f"\n- {col}: {dist.to_dict()}"
    elif question_type == "trend":
        time_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if time_cols:
            col = time_cols[0]
            df_sorted = df.sort_values(by=col)
            summary += f"Sample over time ({col}):\n" + df_sorted[[col]].head(5).to_string(index=False)
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
    else:
        summary += df.describe(include='all').to_string()
    return summary

def convert_nl_to_sql(state: State) -> State:
    question = state["question"]
    system = state["system_prompt"]
    llm = OllamaLLM(model="mistral-nemo:latest", temperature=0.0)
    convert_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {question}"),
    ])
    sql_generator = convert_prompt | llm
    print(f"Converting question to SQL: {question}")
    result = sql_generator.invoke({"question": question})
    message = re.sub(r'^\s*```sql\s*|\s*```$', '', result.strip(), flags=re.IGNORECASE)
    message = re.sub(r"confidence\s*=\s*'high'", "confidence = 'High'", message, flags=re.IGNORECASE)
    message = re.sub(r"confidence\s*=\s*'medium'", "confidence = 'Medium'", message, flags=re.IGNORECASE)
    message = re.sub(r"confidence\s*=\s*'low'", "confidence = 'Low'", message, flags=re.IGNORECASE)
    if "grouped" in message and "supplier" in message and "item.case.supplier" not in message:
        message = re.sub(r"([^.])supplier", r"\1item.case.supplier", message)
    state["sql_query"] = message
    state["attempts"] = 0
    return state

def execute_sql(state: State) -> State:
    db_conn = state["db_conn"]
    query = state["sql_query"]
    error = state.get("sql_error", True)
    result = state.get("query_result", None)
    dataframe = state.get("query_df", None)
    if error or result is None:
        print(f"🚀 Executing query: {query}")
        try:
            allowed_tables = ["cases", "activities", "variants", "grouped", "invoices"]
            if not any(table in query.lower() for table in allowed_tables):
                raise ValueError("Query must target only the allowed tables.")
            cursor = db_conn.cursor()
            cursor.execute(query)
            if query.lower().startswith("select"):
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                formatted_result = "\n".join(
                    ", ".join(f"{col}: {row[idx]}" for idx, col in enumerate(columns)) for row in rows
                ) if rows else "No results found."
                dataframe = pd.DataFrame(rows, columns=columns)
            else:
                formatted_result = "The action has been successfully completed."
            result = formatted_result
            error = False
        except Exception as e:
            result = f"Error executing SQL query: {str(e)}"
            error = True
    state["query_result"] = result
    state["sql_error"] = error
    state["query_df"] = dataframe
    return state

def generate_serious_answer(state: State) -> State:
    question = state["question"]
    query_result = state["query_result"]
    system = f"""
    You are ✨SOFIA✨, an AI business assistant.
    Your task is to:
    1. Answer the user's **main question** using the SQL results from the **sub-questions**.
    2. Provide business insights based on the query results.
    Context: {question}
    SQL Results: {query_result}
    """
    llm = OllamaLLM(model="phi4:latest", temperature=0.0, max_tokens=200)
    response = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", f"Question: {question}"),
    ]) | llm | StrOutputParser()
    state["final_answer"] = response.invoke({})
    return state

def regenerate_query(state: State) -> State:
    error = state["query_result"]
    query = state["sql_query"]
    repair_prompt = state["repair_prompt"]
    llm = OllamaLLM(model="mistral:latest", temperature=0.0)
    print(f"⚠️ Fixing SQL query: {query}")
    print(f"🔍 Error encountered: {error}")
    sql_fix_prompt = ChatPromptTemplate.from_messages([
        ("system", repair_prompt),
        ("human", "Fix the query and return only the corrected SQL, no explanations."),
    ])
    fixer = sql_fix_prompt | llm
    corrected_query = fixer.invoke({"query": query, "error": error})
    corrected_query = re.sub(r"```sql\s*(.*?)\s*```", r"\1", corrected_query.strip(), flags=re.DOTALL | re.IGNORECASE)
    state["sql_query"] = corrected_query
    state["attempts"] += 1
    return state

def summarize_results(state: State) -> State:
    result = state["query_result"]
    dataframe = state["query_df"]
    tokenizer = state["tokenizer"]
    count = count_tokens(result, tokenizer)
    if count <= 2000:
        return state
    question = state["question"]
    question_type = identify_question_type(question)
    summary = f"📊 Summary of result:\n"
    summary += f"- Rows: {len(dataframe)}\n"
    summary += f"- Columns: {', '.join(dataframe.columns)}\n\n"
    summary += f"🔹 Type: {question_type.capitalize()}-based Summary:\n"
    summary += summarize_dataframe(dataframe, question_type)
    state["query_result"] = summary
    return state

def end_max_iterations(state: State) -> State:
    state["query_result"] = "Please try again."
    state["final_answer"] = "I couldn't generate a valid SQL query after 3 attempts. Please try again."
    return state

def check_attempts_router(state: State) -> str:
    return "Retries < 3" if state["attempts"] <= 3 else "Retries >= 3"

def execute_sql_router(state: State) -> str:
    return "Success" if not state["sql_error"] else "Error"

from langgraph.graph import StateGraph

def run_sql_workflow(question, db_conn, tokenizer, context, system_prompt, repair_prompt):
    workflow = StateGraph(State)
    workflow.add_node("Generates SQL queries", convert_nl_to_sql)
    workflow.add_node("Executes SQL", execute_sql)
    workflow.add_node("Regenerate Error-Queries", regenerate_query)
    workflow.add_node("Answer Relevant Question", generate_serious_answer)
    workflow.add_node("Stops due to max Iterations", end_max_iterations)
    workflow.add_node("Summarizes Results", summarize_results)
    workflow.set_entry_point("Generates SQL queries")
    workflow.add_edge("Generates SQL queries", "Executes SQL")
    workflow.add_conditional_edges("Executes SQL", execute_sql_router, {
        "Success": "Summarizes Results",
        "Error": "Regenerate Error-Queries",
    })
    workflow.add_edge("Summarizes Results", "Answer Relevant Question")
    workflow.add_conditional_edges("Regenerate Error-Queries", check_attempts_router, {
        "Retries < 3": "Executes SQL",
        "Retries >= 3": "Stops due to max Iterations",
    })
    workflow.set_finish_point("Answer Relevant Question")
    chain = workflow.compile()
    result = chain.invoke({
        "question": question,
        "db_conn": db_conn,
        "tokenizer": tokenizer,
        "context": context,
        "system_prompt": system_prompt,
        "repair_prompt": repair_prompt
    })
    return result["final_answer"], result["query_result"]

__all__ = ["run_sql_workflow", "run_think_task"]
