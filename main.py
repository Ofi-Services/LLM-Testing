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
from fastapi import FastAPI, Request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


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
    use_case: str


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

## Prompts 
p1_p= """ 
    You are an SQL assistant specialized in DuckDB. Your task is to generate accurate SQL queries based on natural language questions, following the schema and rules below.

    ### Schema (Aliased)

    - **cases**  
    - id (VARCHAR): Case identifier (PK)  
    - avg_time (DOUBLE): Total duration (sec) from start to closure  
    - type, branch, ramo, broker, state, client, creator (VARCHAR): Case metadata  
    - value (BIGINT): Insurance amount  
    - approved (BOOLEAN): Approval status  
    - case_order_date, case_estimated_delivery, case_delivery (TIMESTAMP): Case timestamps  
    - case_employee_id, case_branch, case_supplier (VARCHAR): Case-specific information  
    - case_number_of_items, case_ft_items (INTEGER): Case item details  
    - case_total_price (DOUBLE): Case total price

    - **activities**  
    - id (BIGINT): Activity identifier (PK)  
    - case_id (VARCHAR): Case ID (FK → cases.id)  
    - timestamp (TIMESTAMP): Activity timestamp  
    - name (VARCHAR): Activity name  
    - case_index (BIGINT): Alias of id  
    - tpt (DOUBLE): Duration of the activity in seconds  
    - user, user_type (VARCHAR): User-related info  
    - automatic, rework (BOOLEAN): Activity flags  
    - case_order_date (TIMESTAMP), case_employee_id (VARCHAR), case_branch (VARCHAR), case_supplier (VARCHAR): Case-related data  
    - case_avg_time (DOUBLE): Average time for the case  
    - case_on_time, case_in_full (BOOLEAN): Delivery status flags  
    - case_number_of_items, case_ft_items (INTEGER): Case item counts  
    - case_total_price (DOUBLE): Case total price  
    - case_estimated_delivery, case_delivery (TIMESTAMP): Delivery-related timestamps

    - **variants**  
    - id (BIGINT): Variant ID (PK for path)  
    - activities (VARCHAR[]): Ordered activity names for this path  
    - cases (VARCHAR[]): IDs of cases that followed this path (→ cases.id)  
    - number_cases (BIGINT): Total cases following this variant  
    - percentage (DOUBLE): Percentage of total cases  
    - avg_time (DOUBLE): Avg duration (sec) across cases in this variant

    ### Query Guidelines

    1. Always reference columns with aliases (e.g., c.id, a.case_id)
    2. Use `UNNEST()` in the `FROM` clause to access list fields like v.activities or v.cases. Do not use `UNNEST()` inside expressions like `= ANY(...)`.
    3. When comparing list values (e.g., activity names), first `UNNEST()` the list in a subquery or CTE, then use direct comparison with `TRIM(...)`.
    4. Use `TRIM()` for comparing activity names (e.g., TRIM(a.name) = TRIM(...))
    5. Avoid unnecessary joins or full scans when possible
    6. Convert time differences with `EXTRACT(EPOCH FROM ...)`
    7. Include all non-aggregated columns in `GROUP BY`

    ### Variant Comparison Rules

    - **Most Frequent Path**:  
    Get the variant with the max number_cases:  
    `SELECT * FROM variants WHERE number_cases = (SELECT MAX(number_cases) FROM variants)`

    - **Variant Durations**:  
    Use `avg_time` from `variants` for variant-level durations. Avoid recomputing durations from activity timestamps unless explicitly requested.

    - **Deviations**:  
    All variants with a different `id` from the most frequent one are deviations.  
    When asked for deviation point, just retrieve the full list of activities from the most frequent variant and compare with the other variants.

    - **Activity Durations Along Most Frequent Path**:  
    1. Extract activities from the most frequent variant using `UNNEST(activities)` in the `FROM` clause.
    2. Join with the `activities` table on trimmed name values.
    3. Group by activity name and compute average `tpt`.

    ### Common Pitfall Corrections

    - Never use `UNNEST()` inside `= ANY(...)`. Instead, `UNNEST` in a `FROM` clause or CTE, then join or filter.
    - Avoid using `> ALL(...)` for comparisons. Use `ORDER BY ... LIMIT 1` or `= (SELECT MAX(...))`.
    - When filtering branches or groups with the highest average, use subqueries like:

        ```
        SELECT branch
        FROM cases
        WHERE approved = TRUE
        GROUP BY branch
        ORDER BY AVG(value) DESC
        LIMIT 1
        ```

    - For aggregated stats over filtered groups (e.g., top branches), prefer subqueries or joins with `IN` from pre-identified sets.
    - If no data matches a filter, return `NULL` instead of failing or using over-restrictive filters.
    - When detecting repeated activities on the same day, use:

        ```
        GROUP BY a.case_id, DATE_TRUNC('day', a.timestamp)
        HAVING COUNT(*) > 1
        ```

        Avoid unnecessary joins with `GENERATE_SERIES`.

    ### Output

    - Return **only** the SQL query. No markdown, no tags, no explanation.
    - Never guess values. Always infer based on the data and schema above.
    """
p2_p= """### Database Schema

                - **cases**  
        - id (VARCHAR): Case identifier (PK)  
        - avg_time (DOUBLE): Total duration (sec) from start to closure  
        - type, branch, ramo, broker, state, client, creator (VARCHAR): Case metadata  
        - value (BIGINT): Insurance amount  
        - approved (BOOLEAN): Approval status  
        - case_order_date, case_estimated_delivery, case_delivery (TIMESTAMP): Case timestamps  
        - case_employee_id, case_branch, case_supplier (VARCHAR): Case-specific information  
        - case_number_of_items, case_ft_items (INTEGER): Case item details  
        - case_total_price (DOUBLE): Case total price

        - **activities**  
        - id (BIGINT): Activity identifier (PK)  
        - case_id (VARCHAR): Case ID (FK → cases.id)  
        - timestamp (TIMESTAMP): Activity timestamp  
        - name (VARCHAR): Activity name  
        - case_index (BIGINT): Alias of id  
        - tpt (DOUBLE): Duration of the activity in seconds  
        - user, user_type (VARCHAR): User-related info  
        - automatic, rework (BOOLEAN): Activity flags  
        - case_order_date (TIMESTAMP), case_employee_id (VARCHAR), case_branch (VARCHAR), case_supplier (VARCHAR): Case-related data  
        - case_avg_time (DOUBLE): Average time for the case  
        - case_on_time, case_in_full (BOOLEAN): Delivery status flags  
        - case_number_of_items, case_ft_items (INTEGER): Case item counts  
        - case_total_price (DOUBLE): Case total price  
        - case_estimated_delivery, case_delivery (TIMESTAMP): Delivery-related timestamps

        - **variants**  
        - id (BIGINT): Variant ID (PK for path)  
        - activities (VARCHAR[]): Ordered activity names for this path  
        - cases (VARCHAR[]): IDs of cases that followed this path (→ cases.id)  
        - number_cases (BIGINT): Total cases following this variant  
        - percentage (DOUBLE): Percentage of total cases  
        - avg_time (DOUBLE): Avg duration (sec) across cases in this variant

            **Relations:**
            - "variants"."cases" references "cases"."id", meaning each variant is followed by multiple cases.
            - "variants"."activities" corresponds to the ordered "activities"."name" values for those cases.
            """
p1_i= """
        You are an SQL assistant specialized in DuckDB. Your task is to generate accurate SQL queries based on natural language questions, following the schema and rules below.

        ### Schema (Aliased)

            - **grouped (g)**  
            - group_id (VARCHAR): Unique identifier for each group (PK)  
            - amount_overpaid (BIGINT): Total overpaid amount for the group  
            - itemCount (BIGINT): Number of items in the group  
            - date (VARCHAR): Date of the group  
            - pattern (VARCHAR): Pattern type for the group 'Similar Value','Similar Reference','Exact Match','Similar Date','Similar Vendor','Multiple'
            - open (BOOLEAN): Status of the group (open or closed)  
            - confidence (VARCHAR): Confidence level for detecting the pattern (e.g., "High", "Medium", "Low")  
            - items (STRUCT[]): Array of items within the group, each containing:
                - **id (INTEGER)**: Item identifier (FK → invoices.id)
                - **case (STRUCT)**: Contains case details, such as:
                    - id (VARCHAR): Case identifier  
                    - order_date (VARCHAR): Order date for the case  
                    - employee_id (VARCHAR): Employee ID handling the case  
                    - branch (VARCHAR): Branch handling the case  
                    - supplier (VARCHAR): Supplier associated with the case  
                    - avg_time (DOUBLE): Average time for the case  
                    - estimated_delivery (VARCHAR): Estimated delivery date for the case  
                    - delivery (VARCHAR): Actual delivery date for the case  
                    - on_time (BOOLEAN): Whether the case was delivered on time  
                    - in_full (BOOLEAN): Whether the case was delivered in full  
                    - number_of_items (INTEGER): Number of items in the case  
                    - ft_items (INTEGER): Number of full-time items in the case  
                    - total_price (INTEGER): Total price of the case  
                - date (VARCHAR): Date of the item  
                - unit_price (VARCHAR): Unit price of the item  
                - quantity (INTEGER): Quantity of the item  
                - value (VARCHAR): Value of the item  
                - pattern (VARCHAR): Pattern type for the group 'Similar Value','Similar Reference','Exact Match','Similar Date','Similar Vendor','Multiple'  
                - open (BOOLEAN): Status of the item (open or closed)  
                - group_id (VARCHAR): Group identifier (FK → grouped.group_id)  
                - confidence (VARCHAR): Confidence level for the item’s pattern (e.g., "high", "medium", "low")  
                - description (VARCHAR): Description of the item  
                - payment_method (VARCHAR): Payment method used for the item  
                - pay_date (VARCHAR): Payment date of the item  
                - special_instructions (VARCHAR): Special instructions for the item  
                - accuracy (INTEGER): Accuracy of the item’s data matching

            - **invoices (i)**  
            - id (BIGINT): Invoice identifier (PK)  
            - date (TIMESTAMP_NS): Date and time the invoice was issued  
            - unit_price (VARCHAR): Unit price of the item in the invoice  
            - quantity (BIGINT): Number of items in the invoice  
            - value (VARCHAR): Total value of the invoice  
            - pattern (VARCHAR): Pattern type for the group 'Similar Value','Similar Reference','Exact Match','Similar Date','Similar Vendor','Multiple'
            - open (BOOLEAN): Status of the invoice (open or closed)  
            - group_id (VARCHAR): Group identifier (FK → grouped.group_id)  
            - confidence (VARCHAR): Confidence level for the invoice's pattern (e.g., "High", "Medium", "Low")  
            - description (VARCHAR): Description of the invoice  
            - payment_method (VARCHAR): Method used for payment  
            - pay_date (TIMESTAMP_NS): Date and time the invoice was paid  
            - special_instructions (VARCHAR): Any special instructions for the invoice  
            - accuracy (BIGINT): Accuracy of the invoice's data matching  
            - case_id (VARCHAR): Case identifier associated with the invoice  
            - case_order_date (TIMESTAMP_NS): Order date of the case  
            - case_employee_id (VARCHAR): Employee associated with the case  
            - case_branch (VARCHAR): Branch where the case was handled  
            - case_supplier (VARCHAR): Supplier associated with the case  
            - case_avg_time (DOUBLE): Average time for the case  
            - case_estimated_delivery (TIMESTAMP_NS): Estimated delivery date for the case  
            - case_delivery (TIMESTAMP_NS): Actual delivery date for the case  
            - case_on_time (BOOLEAN): Whether the case was delivered on time  
            - case_in_full (BOOLEAN): Whether the case was delivered in full  
            - case_number_of_items (BIGINT): Number of items in the case  
            - case_ft_items (BIGINT): Number of full-time items in the case  
            - case_total_price (BIGINT): Total price of the case

        ### Query Guidelines

        1. **Prefer Direct Tables**:  
        Use `grouped (g)` or `invoices (i)` directly unless item-level fields are explicitly needed.

        2. **UNNEST Only When Necessary**:
        - Only use `UNNEST(g.items) AS item` when accessing nested fields (e.g., `item.case.supplier`, `item.unit_price`, etc.)
        - After unnesting, access fields as `item.field` or `item.case.supplier`, **not** `item.unnest.field`.

        3. **Nesting and Access Rules**:
        - To access supplier from `grouped`, unnest items and use:  
            ```sql
            FROM grouped g, UNNEST(g.items) AS item
            WHERE item.case.supplier = 'Example'
            ```
        - Avoid referencing nested fields without unnesting first.

        4. **Case Sensitivity**:
        - Use exact case for values:
            - Confidence: 'High', 'Medium', 'Low'
            - Pattern: 'Similar Value', 'Similar Reference', 'Exact Match', 'Similar Date', 'Similar Vendor', 'Multiple'

        5. **Use Table Aliases**:
        - Always use `g.` for `grouped`, `i.` for `invoices`, and `item.` after unnesting.

        6. **Use TRIM() for Comparisons**:
        - For text comparisons like pattern or supplier, wrap with `TRIM()`.  
            Example: `TRIM(item.case.supplier) = 'VendorName'`

        7. **Use IN / = ANY for Multiple Matches**:
        - Use `pattern = ANY (['Value1', 'Value2'])` or `IN (...)` instead of OR chains.

        8. **GROUP BY Nested Fields**:
        - If grouping by nested fields like supplier, first unnest, then group by `item.case.supplier`.

        9. **Aggregation and Filtering**:
        - Use `ORDER BY ... LIMIT 1` instead of `> ALL(...)`
        - Filter early with WHERE clauses to improve performance.

        10. **Alternative Access**:
        - Use `invoices` for simpler flat queries (e.g., `i.case_supplier`).

        ---

        ### Output Rules

        - ❌ Do NOT explain the query.
        - ✅ Only return the SQL query (no markdown, no comments, no formatting).
        - ❌ Do NOT guess field names.
        - ✅ Always respect the provided schema and capitalization.
        """

p2_i= """ 
    ### Schema (Aliased)

    - **grouped (g)**  
    - group_id (VARCHAR): Unique identifier for each group (PK)  
    - amount_overpaid (BIGINT): Total overpaid amount for the group  
    - itemCount (BIGINT): Number of items in the group  
    - date (VARCHAR): Date of the group  
    - pattern (VARCHAR): Pattern type for the group 'Similar Value','Similar Reference','Exact Match','Similar Date','Similar Vendor','Multiple'
    - open (BOOLEAN): Status of the group (open or closed)  
    - confidence (VARCHAR): Confidence level for detecting the pattern (e.g., "High", "Medium", "Low")  
    - items (STRUCT[]): Array of items within the group, each containing:
        - **id (INTEGER)**: Item identifier (FK → invoices.id)
        - **case (STRUCT)**: Contains case details, such as:
            - id (VARCHAR): Case identifier  
            - order_date (VARCHAR): Order date for the case  
            - employee_id (VARCHAR): Employee ID handling the case  
            - branch (VARCHAR): Branch handling the case  
            - supplier (VARCHAR): Supplier associated with the case  
            - avg_time (DOUBLE): Average time for the case  
            - estimated_delivery (VARCHAR): Estimated delivery date for the case  
            - delivery (VARCHAR): Actual delivery date for the case  
            - on_time (BOOLEAN): Whether the case was delivered on time  
            - in_full (BOOLEAN): Whether the case was delivered in full  
            - number_of_items (INTEGER): Number of items in the case  
            - ft_items (INTEGER): Number of full-time items in the case  
            - total_price (INTEGER): Total price of the case  
        - date (VARCHAR): Date of the item  
        - unit_price (VARCHAR): Unit price of the item  
        - quantity (INTEGER): Quantity of the item  
        - value (VARCHAR): Value of the item  
        - pattern (VARCHAR): Pattern type for the group 'Similar Value','Similar Reference','Exact Match','Similar Date','Similar Vendor','Multiple'  
        - open (BOOLEAN): Status of the item (open or closed)  
        - group_id (VARCHAR): Group identifier (FK → grouped.group_id)  
        - confidence (VARCHAR): Confidence level for the item’s pattern (e.g., "high", "medium", "low")  
        - description (VARCHAR): Description of the item  
        - payment_method (VARCHAR): Payment method used for the item  
        - pay_date (VARCHAR): Payment date of the item  
        - special_instructions (VARCHAR): Special instructions for the item  
        - accuracy (INTEGER): Accuracy of the item’s data matching

    - **invoices (i)**  
    - id (BIGINT): Invoice identifier (PK)  
    - date (TIMESTAMP_NS): Date and time the invoice was issued  
    - unit_price (VARCHAR): Unit price of the item in the invoice  
    - quantity (BIGINT): Number of items in the invoice  
    - value (VARCHAR): Total value of the invoice  
    - pattern (VARCHAR): Pattern type for the group 'Similar Value','Similar Reference','Exact Match','Similar Date','Similar Vendor','Multiple'
    - open (BOOLEAN): Status of the invoice (open or closed)  
    - group_id (VARCHAR): Group identifier (FK → grouped.group_id)  
    - confidence (VARCHAR): Confidence level for the invoice's pattern (e.g., "High", "Medium", "Low")  
    - description (VARCHAR): Description of the invoice  
    - payment_method (VARCHAR): Method used for payment  
    - pay_date (TIMESTAMP_NS): Date and time the invoice was paid  
    - special_instructions (VARCHAR): Any special instructions for the invoice  
    - accuracy (BIGINT): Accuracy of the invoice's data matching  
    - case_id (VARCHAR): Case identifier associated with the invoice  
    - case_order_date (TIMESTAMP_NS): Order date of the case  
    - case_employee_id (VARCHAR): Employee associated with the case  
    - case_branch (VARCHAR): Branch where the case was handled  
    - case_supplier (VARCHAR): Supplier associated with the case  
    - case_avg_time (DOUBLE): Average time for the case  
    - case_estimated_delivery (TIMESTAMP_NS): Estimated delivery date for the case  
    - case_delivery (TIMESTAMP_NS): Actual delivery date for the case  
    - case_on_time (BOOLEAN): Whether the case was delivered on time  
    - case_in_full (BOOLEAN): Whether the case was delivered in full  
    - case_number_of_items (BIGINT): Number of items in the case  
    - case_ft_items (BIGINT): Number of full-time items in the case  
    - case_total_price (BIGINT): Total price of the case

"""


prompts_sql_generation= {"0":[p1_p,p2_p],
                         "1":[p1_i,p2_i]}

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
        You are a helpful assistant working for a business intelligence tool. Your job is to determine if a user's message is relevant to the business datasets used by the assistant. The assistant answers only business-related questions that can be answered using SQL queries over structured tables.

        ## When to Consider a Question Relevant
        A question is relevant if it can be answered using data from the database. Relevant questions often involve:
        - Metrics, KPIs, or business values
        - Processes, durations, frequencies, or variant analysis
        - Invoices, vendors, items, dates, payment terms
        - Any reference to structured data the assistant has access to

        A question is considered **relevant** only if it is structured in a way that could be used to extract data from the database.

        ## When a Question is NOT Relevant
        A question is **not relevant** if:
        - It is personal, fictional, or unrelated to the datasets
        - It is vague, humorous, or purely conversational
        - It cannot be answered using the schema below

        Examples of **not relevant** questions:
        - "Hi"
        - "What’s your favorite food?"
        - "Tell me a joke"
        - "Write a haiku about invoices"
        - "How is your day going?"

        ---

        ### Schema Summary

        The database supports two business domains:

        1. **Process Mining Use Case**:
        - Tables: `cases`, `activity`, `variants`
        - Key topics: case durations, activity sequences, variants, average time, brokers, clients, case creators, timestamps, process deviations.

        2. **Duplicate Invoice Checker Use Case**:
        - Tables: `invoices`, `grouped`
        - Key topics: invoice values, vendors, overpaid amounts, delivery delays, duplicate detection patterns (e.g., same vendor and reference), item groups, payment terms, accuracy scores.

        Any user question that targets these types of data is considered relevant.

        ---

        ### Output Format

        Respond only with one of the following:
        - `relevant`
        - `not_relevant`
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
        You are a business-focused assistant specializing in process mining and supplier invoice deduplication analytics.
        Your goal is to interpret ambiguous or complex user questions and convert them into clear, self-contained, measurable prompts for a SQL-capable agent, tailored to the selected use case (process mining or supplier invoice deduplication).

        ### Task 1: Reformat Vague or Indirect Questions
        If the question is vague (e.g., "And the invoices?", "Any duplicates?") or phrased indirectly ("I wonder if..."), rewrite it as a fully clear and self-contained analytical question. Use the context provided in the chat history and the use case (process mining or supplier invoice deduplication).
        - Normalize time expressions such as "last month", "this week", or "recently" into explicit phrases like "in the last 30 days" or "in March 2025".
        - Resolve references like "those", "they", or "that" using context from the chat history, ensuring alignment with the use case (e.g., cases for process mining, duplicate invoices for invoice deduplication).
        - If the question is implicit or easily inferred from the data, do not decompose it into sub-questions.

        ### Task 2: Decompose Multi-Part or Analytical Questions
        If the question contains multiple aspects (e.g., comparisons, multiple KPIs, deviations vs. standard paths for process mining, or duplicate invoice criteria for invoices), decompose it into clear, measurable, self-contained sub-questions.
        - Identify when a question contains comparative logic (e.g., "vs", "compare", "difference between") and split accordingly, considering the use case (e.g., comparing case durations or invoice duplication metrics).
        - For process mining, if the question mentions deviations, identify the reference path (typically the most frequent variant) and ask what diverges from it and where.
        - For invoice deduplication, break questions into steps that examine:
        - Pattern types:
            - 'Exact Match': Identical invoice details.
            - 'Similar Reference': Matching or near-matching reference numbers.
            - 'Similar Vendor': Same or similar suppliers.
            - 'Similar Date', 'Similar Value': Close match in those fields.
            - 'Multiple': Combination of patterns.
        - Confidence:
            - 'High': Confidence ≥ 95%
            - 'Medium'
            - 'Low'
        - Treat invoices with pattern = 'Exact Match' and Confidence = 'High' as **confirmed duplicates**.
        - Treat invoices with Confidence = 'High' (any pattern except 'Exact Match') as **possible duplicates**.
        - When the question asks **"which vendor has the most duplicates"**, split the analysis into:
        - Confirmed duplicates → pattern = 'Exact Match' and Confidence = 'High'.
        - Possible duplicates → Confidence = 'High' and pattern in any of: 'Similar Reference', 'Similar Vendor', 'Similar Value', 'Similar Date', 'Multiple'.

        ### Task 3: Ensure Actionable Metric Framing
        Whenever possible, reframe subjective or abstract queries into questions that can be answered with measurable metrics, tailored to the use case. For example:
        - Process mining: "Is onboarding taking too long?" → "What is the average duration for onboarding cases?"
        - Process mining: "Where are the biggest delays?" → "Which activity has the highest average time between steps?"
        - Supplier invoices: "Are there duplicate invoices?" → 
        - "Which invoices have pattern = 'Exact Match' and Confidence = 'High'?"
        - "Which invoices have Confidence = 'High' with other patterns?"
        - Supplier invoices: "Which supplier has the most duplicated invoices?" → 
        - "Which vendor has the most invoices with pattern = 'Exact Match' and Confidence = 'High'?"
        - "Which vendor has the most invoices with any pattern and Confidence = 'High'?"

        ### Use Case:
        - Process Mining: Questions relate to cases, activities, and variants (e.g., case duration, activity frequency).
        - Supplier Invoice Deduplication: Questions focus on identifying duplicate invoices using:
        - pattern: 'Similar Value', 'Similar Reference', 'Exact Match', 'Similar Date', 'Similar Vendor', 'Multiple' (case-sensitive)
        - confidence: 'High', 'Medium', 'Low' (first-letter capitalized)

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
        # Look for the 'sub_questions' part in the result
        sub_questions_match = re.search(r'"sub_questions"\s*:\s*(\[\s*(.*?)\s*\])', result, re.DOTALL)
        
        if sub_questions_match:
            # Extract and clean up the sub_questions list (remove extra spaces and newlines)
            sub_questions_str = sub_questions_match.group(1)
            # Try to manually fix any missing commas in the sub-questions list
            # Insert commas between question items (if any)
            cleaned_sub_questions = re.sub(r'("\s*[^,"]+)(\s*"\s*[^,"]+)', r'\1,\2', sub_questions_str)
            cleaned_sub_questions = cleaned_sub_questions.replace('",\n "', '", "').replace('\n', ' ').replace('"\n', '"')
            
            try:
                # Attempt to parse the cleaned version of sub_questions
                cleaned_parsed_result = json.loads('{"sub_questions": ' + cleaned_sub_questions + '}')
                state["questions"]= cleaned_parsed_result["sub_questions"]
            except json.JSONDecodeError:
                # In case it still fails, attempt to split based on common delimiters (e.g., '?')
                questions = sub_questions_str.split('?')
                state["questions"]= [q.strip() + '?' for q in questions if q.strip()]

    return state

def select_use_case(state: State):
    """
    Selects the most relevant use case based on the user's question and the database schema.

    Args:
        state (State): The current state of the workflow.

    Returns:
        State: Updated state with the selected use case.
    """
    question = state["original_question"]
    print(f"Selecting use case for question: {question}")

    # System prompt for selecting use cases
    system = """
    You are a classification assistant specialized in understanding user questions related to a database of business processes and financial invoices.
 
    ### Objective:
    Classify each incoming user question into one of two use cases:
    - **Duplicate Invoice Detection (1)** 
    - **Process Mining (0)**
 
    ### Classification Rules:
 
    1. **Duplicate Invoice Detection (Return 1)**:
        - If the question asks about invoices, payment values, unit prices, matching patterns (e.g., "similar vendor", "similar reference", "exact match").
        - If the question involves confidence scores, overpayments, invoice comparisons, amounts, or payment methods.
        - If the question mentions "grouped invoices", "duplicate invoices", "overpaid invoices", "matching errors", or similar.
        - Keywords to look for: `invoice`, `group_id`, `pattern`, `confidence`, `overpaid`, `amount_overpaid`, `similar reference`, `similar vendor`, `payment method`, `unit price`, `value`.
 
    2. **Process Mining (Return 0)**:
        - If the question asks about case flows, activity sequences, event logs, timing between activities, or case durations.
        - If the question analyzes how cases are processed over time, how activities relate, case trends, bottlenecks, process variants, or paths.
        - Keywords to look for: `case`, `activity`, `timestamp`, `workflow`, `process path`, `variant`, `event sequence`, `rework`, `automatic`, `case_id`, `activities list`, `case duration`, `activity name`.
 
    ### Important Notes:
    - Focus on the **main intent** of the question, not just keywords.
    - Even if both invoices and cases are mentioned, classify based on what the user mainly wants to analyze.
    - Respond with **only**:
        - `1` (if duplicate invoice detection)
        - `0` (if process mining)
 
    ### Response Format:
    Return only `1` or `0` without any explanation.
    """


    # Define the human prompt with the user's question
    human = f"Question: {question}"

    # Create a prompt template for the LLM
    select_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )

    # Invoke the LLM to select a use case
    llm = OllamaLLM(model="mistral-nemo:latest", temperature=0.0)
    selector = select_prompt | llm
    response = selector.invoke({}).strip()

    # Update the state with the selected use case
    if "0" in response:
        response = "0"
    else:
        response = "1"
    state["use_case"] = response
    print(f"Selected Use Case: {state['use_case']}")
    return state
    

def convert_nl_to_sql(state: State):
    """
    Converts a natural language question into an SQL query based on the database schema.
 
    Args:
        state (State): The current state of the workflow.
 
    Returns:
        State: Updated state with the generated SQL query.
    """
    questions = state["questions"]
    # Seleccionar el prompt apropiado basado en el caso de uso
    print(state["use_case"])
    system = prompts_sql_generation[state["use_case"]][0]  
    # Agregar información específica sobre case sensitivity y estructura de tablas
    additional_notes = """
    IMPORTANT GUIDELINES:
    1. CASE SENSITIVITY:
    - For confidence values, use 'High', 'Medium', 'Low' (first letter capitalized)
    - For pattern values, use 'Similar Value','Similar Reference','Exact Match','Similar Date','Similar Vendor','Multiple' (with exact capitalization)
    - All string comparisons should respect the exact case of values in the database
    2. NESTED STRUCTURE ACCESS:
    - In the "grouped" table, fields like "supplier" are nested within items.case
    - Correct access pattern: `item.case.supplier` NOT `supplier`
    - When querying supplier information, use:
      * `item.case.supplier` when accessing from unnested items
      * Alternatively, you can use the "invoices" table where supplier is directly accessible as `case_supplier`
    3. DUCKDB UNNEST USAGE:
    - When working with the "grouped" table, use UNNEST to access array elements:
      ```sql
      SELECT item.case.supplier
      FROM grouped g, UNNEST(g.items) AS item
      WHERE ...
      ```
    Always double-check field access paths for nested structures!
    """
    # Añadir las notas adicionales al prompt del sistema
    enhanced_system = system + "\n" + additional_notes
    llm = OllamaLLM(model="mistral-nemo:latest", temperature="0.0")
 
    convert_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", enhanced_system),
                ("human", "Question: {question}"),
            ]
        )
    sql_generator = convert_prompt | llm
    querys = []
    for question in questions:
        print(f"Converting question to SQL {question}")
        result = sql_generator.invoke({"question": question})
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
        querys.append(message)  # Añadir cada consulta SQL generada a la lista
        print(f"Generated SQL query: {message}")
    state["sql_querys"] = querys 
    state["executed"] = [False] * len(state["sql_querys"])  # Inicializar estado de ejecución para cada pregunta
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
    You are ✨SOFIA✨, an AI business assistant. 
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
    llm = OllamaLLM(model="phi4:latest", temperature="0.0", max_tokens=200)
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
    error_state = state["sql_error"]
    error_indexes = [index for index, error in enumerate(error_state) if error == True]

    llm = OllamaLLM(model="mistral:latest", temperature=0.0)
    print(f"🔄 Regenerating query. Attempt {state['attempts'] + 1}")
    for index in error_indexes:
        # Fix the SQL query using the error message
        query = state["sql_querys"][index]
        error = state["query_results"][index]
        print(f"⚠️ Fixing SQL query at index {index}: {query}")
        print(f"🔍 Error encountered: {error}")
        part1= f"""You are an expert in SQL for DuckDB.
            Your task is to correct the following SQL query based on the error message.

            ### **Query to Fix:**
            ```sql
            {query}
            ```

            ### **Error Message:**
            {error}

            Provide a **corrected** SQL query that runs successfully in the following database schema.
            """
        part_2= prompts_sql_generation[state["use_case"]][1]  # Select the appropriate prompt based on use case
        sql_fix_prompt = ChatPromptTemplate.from_messages([(
            "system", 
            part1+part_2),
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

        if count_tokens(result,tokenizer) <= 2000:
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
    system = f"""You are ✨SOFIA✨, a charming and funny assistant. 
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
    if state["attempts"] <= 3:
        print(f"Attempt {state['attempts']}")
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
    a.run()
    db_conn= a.conn
    login(token="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")


    # Create the workflow state
    workflow = StateGraph(State)
    workflow.add_node("Checks Question Relevance", check_relevance)
    workflow.add_node("Selects Use Case", select_use_case)
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
            "Relevant":"Selects Use Case",
            "Not Relevant": "Answer Irrelevant Question"
            } 

        )

    workflow.add_edge("Selects Use Case", "Reformat Question")

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

def message_api(message):
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    a= CargaDeArchivos()
    a.run()
    db_conn= a.conn
    login(token="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")


    # Create the workflow state
    workflow = StateGraph(State)
    workflow.add_node("Checks Question Relevance", check_relevance)
    workflow.add_node("Selects Use Case", select_use_case)
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
            "Relevant":"Selects Use Case",
            "Not Relevant": "Answer Irrelevant Question"
            } 

        )

    workflow.add_edge("Selects Use Case", "Reformat Question")

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
    input_question = message
    
    # Check for exit or goodbye phrases

    # Invoke the chain and ensure chat history persists
    state = chain.invoke({"original_question": input_question, "db_conn": db_conn, "chat_history": chat_history, "tokenizer":tokenizer})
    # Get response and ensure sOFIa is not repeated
    response = state["final_answer"].replace("sOFIa: ", "").strip()
    # Print the response correctly
    print(f"sOFIa: {response}")
    relevance= state["relevance"]
    # Append the interaction to chat history
    chat_history.append(f"User: {input_question} [Relevance: {relevance}]")
    chat_history.append(f"{response}")
    # Get the next question
    # Print the chat history in a well-formatted way
    return chat_history



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O especifica tus dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")

async def create_item(request: Request):
    data = await request.json()  # Accede al payload JSON directamente
    question= data.get("message")
    history= message_api(question)  # Llama a la función message_api con el mensaje del payload
    return {"answer": history[1]}  # Devuelve el historial de chat actualizado como respuesta JSON

