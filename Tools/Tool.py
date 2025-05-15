from langgraph.graph import StateGraph
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from transformers import AutoTokenizer
import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)

# Prompts
from Tools.Prompts import p1_i, p1_p, p2_i, p2_p, system_prompt, classification_prompt

# === AUXILIARY FUNCTIONS === 


def remove_think_tags(text: str) -> str:
    try:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()
    except Exception as e:
        logger.exception(f"Error removing think tags: {e}")
        return text.strip()

# === THINKING  WORKER  ===
#  
def run_think_task(task: str, context: str = "") -> str:
    logger.info("Running run_think_task")
    try:
        llm = OllamaLLM(model="qwen3:8b", temperature=0.0, enable_thinking=False)
        #system_prompt = """ /no_think
        #You are a reasoning engine. Your job is to logically analyze a task, optionally using provided context,
        #and generate a clear, accurate response. Be concise, factual, and business-relevant.
        #"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context: {context}\nTask: {task}"),
        ])
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"task": task, "context": context})
        result = remove_think_tags(result)
        return result
    except Exception as e:
        logger.exception(f"Error during thinking worker: {e}")
        return "An error occurred while processing the task."

# === SQL WORKER ===
# == SQL WORKER STATE ==
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
    use_case: None


# PROMPTS PER USE CASE #
prompts_sql_generation= {"0":[p1_p,p2_p],
            "1":[p1_i,p2_i]}

# == SQL WORKER NODES ==

def count_tokens(text: str, tokenizer) -> int:
    try: 
        length = len(tokenizer.encode(text))
    except Exception as e:
        logger.exception(f"Error counting tokens: {e}")
        length = 0
    return length

def identify_question_type(question: str) -> str:
    try: 
        question = question.lower()
        if any(word in question for word in ["average", "mean", "duration", "time taken", "how long"]):
            return "average"
        if any(word in question for word in ["distribution", "frequency", "histogram"]):
            return "distribution"
        if any(word in question for word in ["trend", "over time", "change", "evolution"]):
            return "trend"
        if any(word in question for word in ["most", "top", "highest", "least", "lowest", "compare"]):
            return "ranking"
        return "general"
    except Exception as e:
        logger.exception(f"Error identifying question type: {e}")
        raise

def summarize_dataframe(df: pd.DataFrame, question_type: str) -> str:
    try:
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
    except Exception as e:
        logger.exception(f"Error summarizing dataframe: {e}")
        return "Error occurred while summarizing the dataframe."
    

def convert_nl_to_sql(state: State) -> State:
    try:
        question = state["question"]
        use_case = state["use_case"]
        system= prompts_sql_generation[use_case][0]
        llm = OllamaLLM(model="qwen3:8b", temperature=0.0, enable_thinking=False)
        convert_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Question: {question}"),
        ])
        sql_generator = convert_prompt | llm
        print(f"Converting question to SQL: {question}")
        result = sql_generator.invoke({"question": question})
        result = remove_think_tags(result)
        message = re.sub(r'^\s*```sql\s*|\s*```$', '', result.strip(), flags=re.IGNORECASE)
        message = re.sub(r"confidence\s*=\s*'high'", "confidence = 'High'", message, flags=re.IGNORECASE)
        message = re.sub(r"confidence\s*=\s*'medium'", "confidence = 'Medium'", message, flags=re.IGNORECASE)
        message = re.sub(r"confidence\s*=\s*'low'", "confidence = 'Low'", message, flags=re.IGNORECASE)
        if "grouped" in message and "supplier" in message and "item.case.supplier" not in message:
            message = re.sub(r"([^.])supplier", r"\1item.case.supplier", message)
        state["sql_query"] = message
        state["attempts"] = 0
        return state
    except Exception as e:
        logger.exception(f"Error converting NL to SQL: {e}")
        raise

# Classification:

## ==Classification Node==
def classify_use_case(state: State) -> State:
    try:
        question = state['question']
        llm = OllamaLLM(model="qwen3:8b", temperature=0.0, enable_thinking=False)
        classifier = ChatPromptTemplate.from_messages([
            ("system", classification_prompt),
            ("human", "{question}")
        ]) | llm | StrOutputParser()

        use_case = classifier.invoke({"question": question}).strip()
        use_case = remove_think_tags(use_case)
        if use_case not in ["0", "1"]:
            logger.exception(f"Not valid use case detected, returning 0 by default: {use_case}")
            use_case = "0"  # by default
        state["use_case"] = use_case
        return state
        
    except Exception as e:
        logger.exception(f"Error during classification node, returning usea case 0 by default: {e}", exc_info=True)
        state["use_case"] = "0"
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
    try:
        question = state["question"]
        print(question)
        query_result = state["query_result"]
        llm = OllamaLLM(model="qwen3:8b", temperature=0.0, max_tokens=200, enable_thinking=False)
        system= f"""
        /no_think
        You are ✨SOFIA✨, an AI business assistant. 
        Your task is to answer the user's question based on the SQL query results.
        Use the SQL query results to support your answer.
        If the SQL query results are empty, indicate you were not able to processe the question.
    
        SQL query results:
        {query_result}
        """
        # print(system)
        model = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", f"Question: {question}"),
        ]) | llm | StrOutputParser()
        response = model.invoke({})
        print(response)
        state["final_answer"] = remove_think_tags(response)
        return state
    except Exception as e:  
        logger.warning(f"Error generating serious answer: {e}")
        state["final_answer"] = "An error occurred while generating the final answer."
        return state

def regenerate_query(state: State) -> State:
    try:

        error = state["query_result"]
        query = state["sql_query"]
        use_case = state["use_case"]
        repair_prompt= prompts_sql_generation[use_case][1]
        
        llm = OllamaLLM(model="qwen3:8b", temperature=0.0, enable_thinking=False)
        print(f"⚠️ Fixing SQL query: {query}")
        print(f"🔍 Error encountered: {error}")
        sql_fix_prompt = ChatPromptTemplate.from_messages([
            ("system", repair_prompt),
            ("human", "Fix the query and return only the corrected SQL, no explanations."),
        ])
        fixer = sql_fix_prompt | llm
        corrected_query = fixer.invoke({"query": query, "error": error})
        corrected_query = remove_think_tags(corrected_query)
        corrected_query = re.sub(r"```sql\s*(.*?)\s*```", r"\1", corrected_query.strip(), flags=re.DOTALL | re.IGNORECASE)
        state["sql_query"] = corrected_query
        state["attempts"] += 1
        return state
    except Exception as e:  
        logger.exception(f"Error regenerating query: {e}")
        state["attempts"] += 1
        return state

def summarize_results(state: State) -> State:
    try:
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
    except Exception as e:
        logger.exception(f"Error summarizing results: {e}")
        state["query_result"] = "An error occurred while summarizing the results."
        return state


def end_max_iterations(state: State) -> State:
    state["query_result"] = "Please try again."
    state["final_answer"] = "I couldn't generate a valid SQL query after 3 attempts. Please try again."
    return state

def check_attempts_router(state: State) -> str:
    return "Retries < 3" if state["attempts"] <= 3 else "Retries >= 3"

def execute_sql_router(state: State) -> str:
    return "Success" if not state["sql_error"] else "Error"


# == SQL WORKER WORFLOW, COMPILE, AND EXECUTE ==
def run_sql_workflow(question, db_conn, tokenizer, context):
    logger.info("Running run_sql_workflow")
    try:
        workflow = StateGraph(State)
        ## Classifier
        workflow.add_node("Classify use case", classify_use_case)
        workflow.add_node("Generates SQL queries", convert_nl_to_sql)
        workflow.add_node("Executes SQL", execute_sql)
        workflow.add_node("Regenerate Error-Queries", regenerate_query)
        workflow.add_node("Answer Relevant Question", generate_serious_answer)
        workflow.add_node("Stops due to max Iterations", end_max_iterations)
        workflow.add_node("Summarizes Results", summarize_results)
        
        workflow.set_entry_point("Classify use case")
        workflow.add_edge('Classify use case', "Generates SQL queries")
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
    except Exception as e:
        logger.exception(f"Error compiling the SQL worker: {e}")
        logger.warning(f"{e}")
        raise
    try:
        result = chain.invoke({
            "question": question,
            "db_conn": db_conn,
            "tokenizer": tokenizer,
            "context": context,
        })
        # print(result["query_result"])
        return result["final_answer"], result["query_result"]
    except Exception as e:
        logger.exception(f"Error executing SQL worker: {e}")
        logger.warning(f"{e}")
        result = {'final_answer':None,
                 'query_result':None}
        # result["final_answer"] = "An error occurred while executing the SQL workflow."
        # result["query_result"] = "An error occurred while executing the SQL workflow."
        # print(result["query_result"])
        return result["final_answer"], result["query_result"]

__all__ = ["run_sql_workflow", "run_think_task", "remove_think_tags"]