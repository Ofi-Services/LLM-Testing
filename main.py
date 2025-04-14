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


class State(TypedDict):
    """
    Represents the state of the workflow, including the question, schema, database connection,
    relevance, SQL query, query result, and other metadata.
    """
    original_question: str
    questions: List[str] = []
    db_conn: None
    query_dfs: List[pd.DataFrame] = []
    relevance: str
    sql_querys: List[str] = []
    query_results: List[str] = []
    sql_error: List[bool]= []
    final_answer: str
    attempts: int
    chat_history: List[str] = []
    tokenizer: None


### Auxiliary functions
def count_tokens(text: str, tokenizer) -> int:
    """
    Count the number of tokens in a given text using the Mistral tokenizer."
    """
    # Tokenize the text and return the number of tokens
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

def relevant_entries(chat_history_entries):
    """
    Filters and retrieves the last 3 relevant user questions and their responses in correct order.

    Args:
        chat_history_entries (list): Full chat history.

    Returns:
        str: A formatted string containing the last 3 relevant interactions in correct order.
    """
    relevant_pairs = []
    found_count = 0
    idx = len(chat_history_entries) - 1

    while idx >= 0:
        entry = chat_history_entries[idx]

        if "[Relevance: relevant]" in entry:
            user_question = entry  # Store user question

            # Look for sOFIa's response **before** storing the question
            response_idx = idx + 1  
            if response_idx < len(chat_history_entries) and chat_history_entries[response_idx].startswith("sOFIa:"):
                sofia_response = chat_history_entries[response_idx]
                relevant_pairs.append((user_question, sofia_response))  # Save as a pair
                found_count += 1

            if found_count >= 3:
                break  # Stop after collecting 3 pairs

        idx -= 1  # Move backwards in history

    # Reverse to maintain chronological order and format correctly
    formatted_history = "\n".join(f"{q}\n{a}" for q, a in reversed(relevant_pairs))
    return formatted_history

def non_relevant_entries(chat_history_entries):
    """
    Filters and retrieves the last 3 non-relevant user questions and their responses in correct order.

    Args:
        chat_history_entries (list): Full chat history.

    Returns:
        str: A formatted string containing the last 3 non-relevant interactions in correct order.
    """
    non_relevant_pairs = []
    found_count = 0
    idx = len(chat_history_entries) - 1

    while idx >= 0:
        entry = chat_history_entries[idx]

        if "[Relevance: not_relevant]" in entry:
            user_question = entry  # Store user question

            # Look for sOFIa's response **before** storing the question
            response_idx = idx + 1  
            if response_idx < len(chat_history_entries) and chat_history_entries[response_idx].startswith("sOFIa:"):
                sofia_response = chat_history_entries[response_idx]
                non_relevant_pairs.append((user_question, sofia_response))  # Save as a pair
                found_count += 1

            if found_count >= 3:
                break  # Stop after collecting 3 pairs

        idx -= 1  # Move backwards in history

    # Reverse to maintain chronological order and format correctly
    formatted_history = "\n".join(f"{q}\n{a}" for q, a in reversed(non_relevant_pairs))
    return formatted_history



## Workflow nodes

def check_relevance(state: State):
    """
    Determines whether the user's question is relevant to the database schema.

    Args:
        state (State): The current state of the workflow.

    Returns:
        State: Updated state with relevance information.
    """
    question = state["original_question"]
    print(f"Checking relevance of the question: {question}")

    # Retrieve chat history
    chat_history_entries = state.get("chat_history", [])
    
    chat_history= relevant_entries(chat_history_entries)  # Get the last 3 relevant entries
    print(f"Chat history for relevance check:\n{chat_history}")
    # System prompt including instructions on chat history usage
    system = f"""
        You are an assistant that determines whether a given question is related to querying the following database schema.
        A question is considered **relevant** only if it is structured in a way that could be used to extract data from the database.
        General conversations, greetings, and small talk are **not relevant**, even if they contain words related to business or databases.
        ---
        ### Infer Implicit References:
        If a question uses **implicit or vague references** like:
        - "Who has the most cases?"
        - "Which one is better?"
        - "How long does it take?"
        ...then:
        **Use the database schema to resolve what "who", "which", or "it" most likely refers to.**
        For example:
        - "Who has the most cases?" → likely refers to **broker** or **creator** in the "cases" table.
        - "Where do most claims happen?" → likely refers to **branch**.
        - "How long does it take?" → likely refers to **durations** (avg_time, insurance_start → insurance_end, etc).
        Assume the **most contextually plausible interpretation** of vague terms.
        ---
        ### How to Use Chat History:
        - If the user's question is a **follow-up to a previous relevant question**, consider it relevant if it maintains the context.
        - If the conversation was **not relevant before**, and the new question is vague or generic, it remains **not relevant**.
        - Do **not** assume relevance unless the question clearly indicates a database query.
        ---
        ### Database Schema  
        #### Table: "cases"
        - "id" (VARCHAR): Primary key.
        - "insurance" (BIGINT): Foreign key to insurance.
        - "avg_time" (DOUBLE): Duration (seconds) from case initiation to closure.
        - "type" (VARCHAR): Insurance category.
        - "branch" (VARCHAR): Policy branch.
        - "ramo" (VARCHAR): Coverage type.
        - "broker" (VARCHAR): Broker for the policy.
        - "state" (VARCHAR): Current case state.
        - "client" (VARCHAR): Client who bought the insurance.
        - "creator" (VARCHAR): Employee managing the case.
        - "value" (BIGINT): Insurance monetary value.
        - "approved" (BOOLEAN): TRUE if approved, else FALSE.
        - "insurance_creation" (TIMESTAMP_NS): Policy creation timestamp.
        - "insurance_start" (TIMESTAMP_NS): Coverage start timestamp.
        - "insurance_end" (TIMESTAMP_NS): Coverage end timestamp.
        #### Table: "activity"
        - "id" (BIGINT): Primary key.
        - "case" (VARCHAR): Foreign key to "cases"."id".
        - "timestamp" (TIMESTAMP_NS): Activity timestamp.
        - "name" (VARCHAR): Name of the activity.
        - "case_index" (BIGINT): Alias for "id".
        - "tpt" (DOUBLE): Activity duration (seconds).
        #### Table: "variants"
        - "id" (BIGINT): Primary key representing a unique variant (i.e., a specific path).
        - "activities" (VARCHAR): Ordered list of activities representing the exact sequence of events in the variant.
        - "cases" (VARCHAR): List of case IDs (as strings) that followed this variant path. Each entry corresponds to a "cases"."id".
        - "number_cases" (BIGINT): Total number of cases that followed this variant.
        - "percentage" (DOUBLE): Proportion (0–100%) of all cases that followed this variant.
        - "avg_time" (DOUBLE): Average total duration (in seconds) of cases following this variant.

        **Relations:**
        - "variants"."cases" contains references to "cases"."id", meaning each variant is followed by multiple cases.
        - "variants"."activities" corresponds to the ordered "activity"."name" values for those cases, showing a summarized path of events.
        ---
        ### Relevance Criteria:
        - A question is **"relevant"** if it queries any concept present in the schema: cases, activities, durations, brokers, clients, values, etc.
        - Business insight questions (e.g., client revenue, broker performance, trends) are relevant.
        - Questions that cannot be reasonably expressed as a query over the schema are **"not_relevant"**.
        - Casual language, greetings, jokes, or unrelated topics are **"not_relevant"**.
        ---
        ### Chat History (Last Exchanges):  
        {chat_history}
        ---
        ### Response Format (STRICT)
        - Respond with exactly one word:
          - `"relevant"`
          - `"not_relevant"`
        Do **not** include explanations or additional commentary.
        """
    
    # Define the human prompt with the user's question
    human = f"Question: {question}"

    # Create a prompt template for the LLM
    check_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )

    # Invoke the LLM to determine relevance
    llm = OllamaLLM(model="mistral:latest", temperature=0.0)
    relevance_checker = check_prompt | llm
    response = relevance_checker.invoke({}).strip().lower()

    # Validate the response to ensure it matches expected outputs
    if response not in ["relevant", "not_relevant"]:
        raise ValueError(f"Unexpected relevance response: {response}")

    # Update the state with the relevance result
    state["relevance"] = response
    state["attempts"] = 0
    print(f"Relevance determined: {state['relevance']}")
    return state



def reformat_question(state: State):
    """
    Reformats vague follow-ups to be self-contained and 
    decomposes complex questions into fully-contained sub-questions.
    
    Args:
        state (Dict): Current workflow state.

    Returns:
        Dict: Updated state with a structured question output.
    """
    original_question = state["original_question"]
    # Retrieve chat history
    chat_history_entries = state.get("chat_history", [])
    
    chat_history= relevant_entries(chat_history_entries)  # Get the last 3 relevant entries

    system_prompt = """
    You are a business-focused assistant specializing in process mining and operational analytics.
    Your goal is to interpret ambiguous or complex user questions and convert them into clear, self-contained, measurable prompts for a SQL-capable agent.
 
    ### Task 1: Reformat Vague or Indirect Questions
    If the question is vague (e.g., "And the cases?", "Those approvals?") or phrased indirectly ("I wonder if..."), rewrite it as a fully clear and self-contained analytical question. Use the context provided in the chat history.
    - Normalize time expressions such as "last month", "this week", or "recently" into explicit phrases like "in the last 30 days" or "in March 2025".
    - Resolve references like "those", "they", or "that" using context from the chat history.
    - If the question is implicit or easily inferred from the data, do not decompose it into sub-questions.
 
    ### Task 2: Decompose Multi-Part or Analytical Questions
    If the question contains multiple aspects (e.g., comparisons, multiple KPIs, deviations vs. standard paths), decompose it into clear, measurable, self-contained sub-questions.
    - Identify when a question contains comparative logic (e.g., "vs", "compare", "difference between") and split accordingly.
    - If the question mentions deviations, identify the reference path (typically the most frequent variant) and ask what diverges from it and where.
    - For process impact questions, break them into steps that examine timing, frequency, and position in the process.
 
    ### Task 3: Ensure Actionable Metric Framing
    Whenever possible, reframe subjective or abstract queries into questions that can be answered with measurable metrics. For example:
    - "Is onboarding taking too long?" → "What is the average duration for onboarding cases?"
    - "Where are the biggest delays?" → "Which activity has the highest average time between steps?"
 
    ### Chat History (for context resolution):
    {chat_history}
 
    **Response Format:**
    If the question is already clear and singular, return it unchanged.
    If it requires decomposition or clarification, return in JSON:
 
    {{
      "sub_questions": ["First rephrased question", "Second one", ...]
    }}
    """
 
    llm = OllamaLLM(model="mistral:latest", temperature=0.1)  

    reformat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User's question: {question}"),
        ]
    )

    reformatter = reformat_prompt | llm
    result = reformatter.invoke({"question": original_question, "chat_history": chat_history})

    # Try parsing JSON if decomposition is detected
    try:
        import json
        parsed_result = json.loads(result)
        if "sub_questions" in parsed_result:
            state["questions"] = parsed_result["sub_questions"]  # Store list of sub-questions
        else:
            state["questions"] = result.strip()  # Store single reformatted question
    except json.JSONDecodeError:
        state["questions"] = result.strip()  # Store as plain text if no JSON structure

    print(f"Processed Question(s): {state['questions']}")

    return state

     

def convert_nl_to_sql(state: State):
    """
    Converts a natural language question into an SQL query based on the database schema.

    Args:
        state (State): The current state of the workflow.
        config (RunnableConfig): Configuration for the runnable.

    Returns:
        State: Updated state with the generated SQL query.
    """
    questions = state["questions"]
    ##PROMPT FOR SQL GENERATION
    system= """
    You are an SQL assistant specialized in DuckDB. Your task is to generate accurate SQL queries based on natural language questions, following the schema and rules below.

    ### Schema (Aliased)

    - **cases (c)**  
    - id (VARCHAR): Case identifier (PK)  
    - insurance (BIGINT): Insurance policy (FK)  
    - avg_time (DOUBLE): Total duration (sec) from start to closure  
    - type, branch, ramo, broker, state, client, creator (VARCHAR): Case metadata  
    - value (BIGINT): Insurance amount  
    - approved (BOOLEAN): Approval status  
    - insurance_creation, insurance_start, insurance_end (TIMESTAMP_NS): Policy timestamps  

    - **activity (a)**  
    - id (BIGINT): Activity identifier (PK)  
    - case (VARCHAR): Case ID (FK → cases.id)  
    - timestamp (TIMESTAMP_NS): Activity timestamp  
    - name (VARCHAR): Activity name  
    - case_index (BIGINT): Alias of id  
    - tpt (DOUBLE): Duration of the activity in seconds  

    - **variants (v)**  
    - id (BIGINT): Variant ID (PK for path)  
    - activities (LIST<TEXT>): Ordered activity names for this path  
    - cases (LIST<VARCHAR>): IDs of cases that followed this path (→ cases.id)  
    - number_cases (BIGINT): Total cases following this variant  
    - percentage (DOUBLE): percentage of total cases  
    - avg_time (DOUBLE): Avg duration (sec) across cases in this variant  

    ### Query Guidelines

    1. Use table aliases: cases → c, activity → a, variants → v  
    2. Always reference columns with aliases (e.g., c.id, a.case)  
    3. Use `UNNEST()` for accessing list fields like v.activities or v.cases  
    4. Use `= ANY(...)` when filtering against list fields  
    5. Use `TRIM()` for comparing activity names (e.g., TRIM(a.name))  
    6. Avoid unnecessary joins or full scans when possible  
    7. Convert time differences with `EXTRACT(EPOCH FROM ...)`  
    8. Include all non-aggregated columns in `GROUP BY`  

    ### Variant Comparison Rules

    - **Most Frequent Path**:  
    Get the variant with the max number_cases:  
    `SELECT * FROM variants WHERE number_cases = (SELECT MAX(number_cases) FROM variants)`

    -**Variant Durations**:
    - Use avg_time from variants when querying variant-level durations. Avoid recomputing durations using activity.timestamp unless explicitly asked.

    - **Deviations**:  
    All variants with a different ID from the most frequent one are deviations.  
    Use `UNNEST(activities)` and compare step-by-step if needed.

    - **Activity Durations Along Most Frequent Path**:  
    1. Extract activities of the most frequent variant  
    2. Join with activity table to calculate avg tpt per activity
    
    ### Common Pitfall Corrections

    - Avoid using `> ALL(...)` for comparisons across groups. Instead, use `ORDER BY ... LIMIT 1` or `= (SELECT MAX(...))` for selecting top values.
    - When filtering branches or groups with the highest average, use subqueries like:

        ```
        SELECT branch
        FROM cases
        WHERE approved = TRUE
        GROUP BY branch
        ORDER BY AVG(value) DESC
        LIMIT 1
        ```

    - When calculating sums/averages for filtered groups (like top branches), prefer joins or IN subqueries from pre-identified groups.

    - If no data matches a filter, return NULL. Do not create over-restrictive filters.

    - When filtering by repeated activities on the same day, use GROUP BY a.case, DATE_TRUNC('day', a.timestamp) and HAVING COUNT(*) > 1. Avoid joining with GENERATE_SERIES.

    ### Output

    - Return **only** the SQL query. No markdown, no tags, no explanation.
    - Never guess values. Always infer based on the data and schema above.
    """


    llm= OllamaLLM(model="mistral-nemo:latest",temperature="0.0")

    convert_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Question: {question}"),
            ]
        )
    
    sql_generator = convert_prompt | llm
    querys= []
    for question in questions:
        print(f"Converting question to SQL {question}")
        result = sql_generator.invoke({"question": question})
        message= re.sub(r'^\s*```sql\s*|\s*```$', '', result.strip(), flags=re.IGNORECASE)
        querys.append(message) # Append each generated SQL query to the list
        print(f"Generated SQL query: {message}")
    state["sql_querys"] = querys 
    state["executed"]= [False] * len(state["sql_querys"])  # Initialize executed status for each question # Store the list of SQL queries in the state
    print(f"Generated SQL queries: {state['sql_querys']}")
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
    sql_queries = state["sql_querys"]
    errors = state.get("sql_error", [True] * len(sql_queries))  # Default: all True (assume they need execution)
    results = state.get("query_results", [None] * len(sql_queries))
    dataframes = state.get("query_dfs", [None] * len(sql_queries))
    for i, query in enumerate(sql_queries):
        if errors[i] or results[i] is None:  # Execute if error OR never executed before
            print(f"🚀 Executing query {i}: {query}")
            try:
                # Ensure the query targets only the allowed tables
                allowed_tables = ["cases", "activity","variants"]
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

                    state["query_rows"] = rows
                    df = pd.DataFrame(rows, columns=columns)
                    dataframes[i] = df  # Store the DataFrame in the state
                else:
                    formatted_result = "The action has been successfully completed."
                    print("SQL command executed successfully.")

                results[i]= formatted_result
                errors[i]= False # Mark this query as executed successfully

            except Exception as e:
                results[i]=f"Error executing SQL query: {str(e)}" # Store the error message in the results
                errors[i]= True # Mark this query as executed with an error
                print(f"Error executing SQL query: {str(e)}")
    state["query_results"] = results  # Store the list of query results in the state
    state["sql_error"] = errors  # Store the list of error states in the state
    state["query_dfs"] = dataframes  # Store the list of DataFrames in the state
    print(f"SQL query results: {state['query_results']}")
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
    question = state["original_question"]
    sub_questions = state["questions"]
    query_results = state["query_results"]  # This is now a list of results, one per sub-question

    chat_history_entries = state.get("chat_history", [])
    chat_history = relevant_entries(chat_history_entries)  # Get the last 3 relevant entries

    # Concatenate each sub-question with its answer
    sub_q_results_str = "\n".join(
        f"**{sq}**\n{qr}\n" for sq, qr in zip(sub_questions, query_results)
    )

    system = f"""
    You are sOFIa, an AI assistant designed by the AI dream team at OFI Services. 
    Your task is to:
    1. Answer the user's **main question** using the SQL results from the **sub-questions**.
    2. Provide business insights based on the query results.

    ### **Chat History:**  
    {chat_history}

    ### **Context:**  
    - **User's Main Question:** {question}  
    - **SQL Results from Sub-Questions:**  
    {sub_q_results_str}

    ### **Instructions:**  
    - Summarize the SQL results in a **clear business-oriented answer**.
    - Every duration is given in seconds, if the number is too high, convert it to minutes or hours.
    - Ensure the answer **directly addresses the main question**.
    - Provide **business insights** based on patterns, trends, and potential improvements.
    - If relevant, compare values or suggest actions based on findings.

    ### **Response Format:**
    - Always return the answer with markdown formatting.
    - Use bullet points for clarity and organization.
    - Avoid excessive jargon; keep it understandable for a business audience.
    - Provide actionable insights or recommendations where applicable.
    - Be careful with the time conversions, and ensure they are accurate.
    """


    human_message = f"Question: {question}"
    
    # Use sOFIa to generate a response based on the SQL result
    llm = OllamaLLM(model="mistral:latest", temperature="0.0", max_tokens=200)
    response = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human_message),
    ]) | llm | StrOutputParser()
    
    # Generate and store the response
    message = response.invoke({})
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
    error_state= state["sql_error"]
    error_indexes = [index for index, error in enumerate(error_state) if error == True]

    llm = OllamaLLM(model="mistral:latest", temperature=0.0)  
    print(f"🔄 Regenerating query. Attempt {state['attempts'] + 1}")
    for index in error_indexes:
        # Fix the SQL query using the error message
        query = state["sql_querys"][index]
        error = state["query_results"][index]
        print(f"⚠️ Fixing SQL query at index {index}: {query}")
        print(f"🔍 Error encountered: {error}")

        # Dynamic Prompt Generation
        sql_fix_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert in SQL for DuckDB.
            Your task is to correct the following SQL query based on the error message.

            ### **Query to Fix:**
            ```sql
            {query}
            ```

            ### **Error Message:**
            {error}

            Provide a **corrected** SQL query that runs successfully in the following database schema.
            ### Database Schema  
            #### Table: "cases"
            - "id" (VARCHAR): Primary key.
            - "insurance" (BIGINT): Foreign key to insurance.
            - "avg_time" (DOUBLE): Duration (seconds) from case initiation to closure.
            - "type" (VARCHAR): Insurance category.
            - "branch" (VARCHAR): Policy branch.
            - "ramo" (VARCHAR): Coverage type.
            - "broker" (VARCHAR): Broker for the policy.
            - "state" (VARCHAR): Current case state.
            - "client" (VARCHAR): Client who bought the insurance.
            - "creator" (VARCHAR): Employee managing the case.
            - "value" (BIGINT): Insurance monetary value.
            - "approved" (BOOLEAN): TRUE if approved, else FALSE.
            - "insurance_creation" (TIMESTAMP_NS): Policy creation timestamp.
            - "insurance_start" (TIMESTAMP_NS): Coverage start timestamp.
            - "insurance_end" (TIMESTAMP_NS): Coverage end timestamp.

            #### Table: "activity"
            - "id" (BIGINT): Primary key.
            - "case" (VARCHAR): Foreign key to "cases"."id".
            - "timestamp" (TIMESTAMP_NS): Activity timestamp.
            - "name" (VARCHAR): Name of the activity.
            - "case_index" (BIGINT): Alias for "id".
            - "tpt" (DOUBLE): Activity duration (seconds).

            #### Table: "variants"
            - "id" (BIGINT): Primary key representing a unique variant (i.e., a specific path).
            - "activities" (VARCHAR): Ordered list of activities representing the exact sequence of events in the variant.
            - "cases" (VARCHAR): List of case IDs (as strings) that followed this variant path. Each entry corresponds to a "cases"."id".
            - "number_cases" (BIGINT): Total number of cases that followed this variant.
            - "percentage" (DOUBLE): Proportion (0–100%) of all cases that followed this variant.
            - "avg_time" (DOUBLE): Average total duration (in seconds) of cases following this variant.

            **Relations:**
            - "variants"."cases" contains references to "cases"."id", meaning each variant is followed by multiple cases.
            - "variants"."activities" corresponds to the ordered "activity"."name" values for those cases, showing a summarized path of events.
                    """),
            ("human", "Fix the query and return only the corrected SQL, no explanations."),
        ])

        fixer = sql_fix_prompt | llm 
        # Pass the query and error message to the SQL model for correction
        corrected_query = fixer.invoke({"query": query, "error": error})
        # Extract only the SQL code from a markdown block like ```sql ... ```
        corrected_query = re.sub(r"```sql\s*(.*?)\s*```", r"\1", corrected_query.strip(), flags=re.DOTALL | re.IGNORECASE)

        state["sql_querys"][index] = corrected_query
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
    query_results = state.get("query_results", [])
    dataframes = state.get("query_dfs", [])
    questions = state.get("questions", [])
    tokenizer= state["tokenizer"]
    for i, result in enumerate(query_results):
        if not result or i >= len(dataframes):
            continue

        if count_tokens(result,tokenizer) <= 1000:
            continue

        df = dataframes[i]
        question = questions[i] if i < len(questions) else ""
        question_type = identify_question_type(question)

        summary = f"📊 Summary of result #{i}:\n"
        summary += f"- Rows: {len(df)}\n"
        summary += f"- Columns: {', '.join(df.columns)}\n\n"
        summary += f"🔹 Type: {question_type.capitalize()}-based Summary:\n"
        summary += summarize_dataframe(df, question_type)

        state["query_results"][i] = summary
        print(f"✅ Summarized result #{i} ({question_type} type, >1000 tokens)")
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
    state["query_results"] = "Please try again."
    state["final_answer"] = "I couldn't generate a valid SQL query after 3 attempts. Please try again."
    print("Maximum attempts reached. Ending the workflow.")
    return state



def generate_funny_response(state: State):
    """
    Generates a playful and humorous response for unrelated questions.
    
    Args:
        state (State): The current state of the workflow.
        
    Returns:
        State: Updated state with the funny response.
    """
    print("Generating a funny response for an unrelated question.")
    question = state["original_question"]
    chat_history_entries = state.get("chat_history", [])
    chat_history = non_relevant_entries(chat_history_entries) # Get the last 3 non-relevant entries
    print(f"Chat history for funny response:\n{chat_history}")
    system = f"""You are **sOFIa**, a charming and funny assistant designed by the AI team at OFI Services. 
    You respond in a playful and lighthearted manner. Your responses should always be fun, engaging, and humorous. 
    If the user doesn't know you yet, introduce yourself!
    
    ### **Chat History:**  
    {chat_history}
    """

    human_message = f"Question: {question}"

    # Generate the playful response
    funny_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human_message),
        ]
    )
    
    llm = OllamaLLM(model="mistral:latest", temperature="0.7",max_tokens=200)
    funny_response = funny_prompt | llm | StrOutputParser()
    message = funny_response.invoke({})
    state["final_answer"] = message
    return state


## Routings
def check_attempts_router(state: State):
    """
    Routes the workflow based on the number of attempts made to generate a valid SQL query.

    Args:
        state (State): The current state of the workflow.

    Returns:
        str: The next node in the workflow.
    """
    if state["attempts"] < 3:
        return "Retries < 3"
    else:
        error_state= state["sql_error"]
        for error in error_state:
            if error == False:
                return "If at least 1 subquery was succesful"
        return "Retries >= 3"



def execute_sql_router(state: State):
    """
    Routes the workflow based on whether the SQL query execution was successful.

    Args:
        state (State): The current state of the workflow.

    Returns:
        str: The next node in the workflow.
    """
    error_state= state["sql_error"]
    for error in error_state:
        if error == True:
            return "Error"
    else:
        return "Success"


    
def relevance_router(state: State):
    """
    Routes the workflow based on the relevance of the user's question.

    Args:
        state (State): The current state of the workflow.

    Returns:
        str: The next node in the workflow.
    """
    if state["relevance"].lower() == "relevant":
        return "Relevant"
    else:
        return "Not Relevant"
    




### Main function to run and create the workflow

def main():

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    a= CargaDeArchivos()
    a.run_carga()
    db_conn= a.conn
    login(token="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")


    # Create the workflow state
    workflow = StateGraph(State)
    workflow.add_node("Checks Question Relevance", check_relevance)
    workflow.add_node("Reformat Question", reformat_question)
    workflow.add_node("Generates SQL queries", convert_nl_to_sql)
    workflow.add_node("Executes SQL",execute_sql)
    workflow.add_node("Regenerate Error-Queries",regenerate_query)
    workflow.add_node("Answer Irrelevant Question", generate_funny_response)
    workflow.add_node("Answer Relevant Question",generate_serious_answer)
    workflow.add_node("Stops due to max Iterations",end_max_iterations)
    workflow.add_node("Summarizes Results", summarize_results)

    workflow.add_edge(START, "Checks Question Relevance")


    workflow.add_conditional_edges(
            "Checks Question Relevance",
            relevance_router,
            {
            "Relevant":"Reformat Question",
            "Not Relevant": "Answer Irrelevant Question"
            } 

        )
    workflow.add_edge("Reformat Question", "Generates SQL queries")

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
                "If at least 1 subquery was succesful": "Summarizes Results",
            },
        )
    workflow.add_edge("Stops due to max Iterations", END)
    workflow.add_edge("Answer Relevant Question",END)
    workflow.add_edge("Answer Irrelevant Question",END)

    chain= workflow.compile()

    print("Hi hi! I'm sOFIa, your assistant!")
    print("Let's get started by asking a question!")
    chat_history = []  # Store chat history outside the loop
    input_question = input()
    while input_question:
        # Check for exit or goodbye phrases
        if input_question.lower() in ["no", "exit", "goodbye", "quit"]:
            print("Goodbye! Have a great day!")
            break
        # Invoke the chain and ensure chat history persists
        state = chain.invoke({"original_question": input_question, "db_conn": db_conn, "chat_history": chat_history, "tokenizer":tokenizer})
        # Get response and ensure sOFIa is not repeated
        response = state["final_answer"].replace("sOFIa: ", "").strip()
        # Print the response correctly
        print(f"sOFIa: {response}")
        relevance= state["relevance"]
        # Append the interaction to chat history
        chat_history.append(f"User: {input_question} [Relevance: {relevance}]")
        chat_history.append(f"sOFIa: {response}")
        # Get the next question
        input_question = input()
    # Print the chat history in a well-formatted way
    print("\nChat History:")
    for entry in chat_history:
        print(entry)

main()