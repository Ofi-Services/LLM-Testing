import os
import pandas as pd
import json
import duckdb
import ast
from datetime import datetime
import logging





class CargaDeArchivos:
    def __init__(self):
        self.data_folder = "./Data"
        self.conn = duckdb.connect(":memory:")
        self.chunksize = 10000  # Adjust based on your memory constraints
        self.logger = logging.getLogger(__name__)
    

    def load_json_in_chunks(self, filename):
        """Load JSON file in chunks to manage memory"""
        try:
            filepath = os.path.join(self.data_folder, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    results = data
                else:
                    results = data.get("results", [])

                self.logger.info(f"Loaded {len(results)} records from {filename}")

                for i in range(0, len(results), self.chunksize):
                    chunk = results[i:i + self.chunksize]
                    yield pd.DataFrame(chunk)
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            yield pd.DataFrame()

    def process_cases(self):
        self.logger.info("Processing cases...")
        try:
            self.conn.execute("DROP TABLE IF EXISTS cases")
            first_chunk = True
            for chunk_df in self.load_json_in_chunks("Case.json"):
                if chunk_df.empty:
                    continue

                datetime_cols = ["order_date", "estimated_delivery", "delivery"]
                for col in datetime_cols:
                    if col in chunk_df.columns:
                        chunk_df[col] = pd.to_datetime(chunk_df[col], utc=True).dt.tz_convert(None)

                type_conversions = {
                    "id": "string", "employee_id": "string", "branch": "string", "supplier": "string",
                    "avg_time": "float64", "on_time": "boolean", "in_full": "boolean",
                    "number_of_items": "Int32", "ft_items": "Int32", "total_price": "float64",
                    "total_activities": "Int32", "rework_activities": "Int32", "automatic_activities": "Int32"
                }
                for col, dtype in type_conversions.items():
                    if col in chunk_df.columns:
                        chunk_df[col] = chunk_df[col].astype(dtype)

                self.conn.register("temp_cases", chunk_df)
                if first_chunk:
                    self.conn.execute("""
                        CREATE TABLE cases AS
                        SELECT * FROM temp_cases
                    """)
                    first_chunk = False
                else:
                    self.conn.execute("""
                        INSERT INTO cases SELECT * FROM temp_cases
                    """)
                self.conn.execute("DROP VIEW temp_cases")
     
        except Exception as e:
            self.logger.error(f"Error processing cases: {e}")

    def process_activities(self):
        self.logger.info("Processing activities...")
        try:
            self.conn.execute("DROP TABLE IF EXISTS activities")
            first_chunk = True
            for chunk_df in self.load_json_in_chunks("Activity.json"):
                if chunk_df.empty:
                    continue

                if "case" in chunk_df.columns:
                    case_df = pd.json_normalize(chunk_df["case"])
                    case_df.columns = [f"case_{col}" for col in case_df.columns]
                    chunk_df = pd.concat([chunk_df.drop(columns=["case"]), case_df], axis=1)

                for col in ["timestamp", "case_order_date", "case_estimated_delivery", "case_delivery"]:
                    if col in chunk_df.columns:
                        chunk_df[col] = pd.to_datetime(chunk_df[col], utc=True).dt.tz_convert(None)

                type_conversions = {
                    "id": "INTEGER", "timestamp": "TIMESTAMP", "name": "VARCHAR", "tpt": "DOUBLE",
                    "user": "VARCHAR", "user_type": "VARCHAR", "automatic": "BOOLEAN", "rework": "BOOLEAN",
                    "case_index": "INTEGER", "case_id": "VARCHAR", "case_order_date": "TIMESTAMP",
                    "case_employee_id": "VARCHAR", "case_branch": "VARCHAR", "case_supplier": "VARCHAR",
                    "case_avg_time": "DOUBLE", "case_estimated_delivery": "TIMESTAMP",
                    "case_delivery": "TIMESTAMP", "case_on_time": "BOOLEAN", "case_in_full": "BOOLEAN",
                    "case_number_of_items": "INTEGER", "case_ft_items": "INTEGER", "case_total_price": "DOUBLE"
                }

                self.conn.register("temp_activities", chunk_df)
                if first_chunk:
                    columns_def = ",\n".join(f"{col} {dtype}" for col, dtype in type_conversions.items())
                    self.conn.execute(f"CREATE TABLE activities ({columns_def})")
                    self.conn.execute("INSERT INTO activities SELECT * FROM temp_activities")
                    first_chunk = False
                else:
                    self.conn.execute("INSERT INTO activities SELECT * FROM temp_activities")
                self.conn.execute("DROP VIEW temp_activities")

        except Exception as e:
            self.logger.error(f"Error processing activities: {e}")

    def process_variants(self):
        self.logger.info("Processing variants...")
        try:
            self.conn.execute("DROP TABLE IF EXISTS variants")
            first_chunk = True
            for chunk_df in self.load_json_in_chunks("Variant.json"):
                if chunk_df.empty:
                    continue

                list_cols = ["activities", "cases"]
                for col in list_cols:
                    if col in chunk_df.columns:
                        chunk_df[col] = chunk_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

                self.conn.register("temp_variants", chunk_df)
                if first_chunk:
                    self.conn.execute("CREATE TABLE variants AS SELECT * FROM temp_variants")
                    first_chunk = False
                else:
                    self.conn.execute("INSERT INTO variants SELECT * FROM temp_variants")
                self.conn.execute("DROP VIEW temp_variants")

        except Exception as e:
            self.logger.error(f"Error processing variants: {e}")

    def process_grouped(self):
        self.logger.info("Processing grouped...")
        try:
            self.conn.execute("DROP TABLE IF EXISTS grouped")
            first_chunk = True
            for chunk_df in self.load_json_in_chunks("Grouped.json"):
                if chunk_df.empty:
                    continue

                self.conn.register("temp_grouped", chunk_df)
                if first_chunk:
                    self.conn.execute("CREATE TABLE grouped AS SELECT * FROM temp_grouped")
                    first_chunk = False
                else:
                    self.conn.execute("INSERT INTO grouped SELECT * FROM temp_grouped")
                self.conn.execute("DROP VIEW temp_grouped")

        except Exception as e:
            self.logger.error(f"Error processing grouped: {e}")

    def process_invoices(self):
        self.logger.info("Processing invoices...")
        try:
            self.conn.execute("DROP TABLE IF EXISTS invoices")
            first_chunk = True
            for chunk_df in self.load_json_in_chunks("Invoice.json"):
                if chunk_df.empty:
                    continue

                if "case" in chunk_df.columns:
                    case_df = pd.json_normalize(chunk_df["case"])
                    case_df.columns = [f"case_{col}" for col in case_df.columns]
                    chunk_df = pd.concat([chunk_df.drop(columns=["case"]), case_df], axis=1)

                datetime_cols = ["date", "pay_date", "case_order_date", "case_estimated_delivery", "case_delivery"]
                for col in datetime_cols:
                    if col in chunk_df.columns:
                        chunk_df[col] = pd.to_datetime(chunk_df[col], utc=True).dt.tz_convert(None)

                self.conn.register("temp_invoices", chunk_df)
                if first_chunk:
                    self.conn.execute("CREATE TABLE invoices AS SELECT * FROM temp_invoices")
                    first_chunk = False
                else:
                    self.conn.execute("INSERT INTO invoices SELECT * FROM temp_invoices")
                self.conn.execute("DROP VIEW temp_invoices")

        except Exception as e:
            self.logger.error(f"Error processing invoices: {e}")

    def inspect_database(self):
        self.logger.info("Inspecting database...")
        try:
            print("\nDatabase Inspection:")
            tables = self.conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
            """).fetchall()

            for (table,) in tables:
                print(f"\n=== {table.upper()} ===")

                print("\nStructure:")
                structure = self.conn.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table}'
                """).df()
                print(structure.to_string(index=False))

                print("\nSample Data (5 rows):")
                sample = self.conn.execute(f"SELECT * FROM {table} LIMIT 5").df()
                print(sample.to_string(index=False))
        except Exception as e:
            self.logger.error(f"Error inspecting database: {e}")

    def run(self):
        self.logger.info("---=== POPULATE DATABASE ===---")
        self.process_cases()
        self.process_activities()
        self.process_variants()
        self.process_grouped()
        self.process_invoices()
        # self.inspect_database()
        self.logger.info("---=== POPULATE DATABE COMPLETED ===---")


if __name__ == "__main__":
    loader = CargaDeArchivos()
    loader.run()
