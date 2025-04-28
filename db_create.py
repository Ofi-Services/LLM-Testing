import os
import pandas as pd
import json
import duckdb
import ast
from datetime import datetime

class CargaDeArchivos:
    def __init__(self):
        self.data_folder = "./Data"
        self.conn = duckdb.connect(":memory:")
        self.chunksize = 10000  # Adjust based on your memory constraints

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
                
                # Process in chunks
                for i in range(0, len(results), self.chunksize):
                    chunk = results[i:i + self.chunksize]
                    yield pd.DataFrame(chunk)
                    
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            yield pd.DataFrame()  # Yield empty DataFrame on error


    def process_cases(self):
        """Process Case.json in chunks and load into DuckDB with proper schema."""
        print("Loading cases data in chunks...")
        self.conn.execute("DROP TABLE IF EXISTS cases")

        first_chunk = True

        for chunk_df in self.load_json_in_chunks("Case.json"):
            if chunk_df.empty:
                continue

            # Convert datetime fields
            datetime_cols = ["order_date", "estimated_delivery", "delivery"]
            for col in datetime_cols:
                if col in chunk_df.columns:
                    chunk_df[col] = pd.to_datetime(chunk_df[col], utc=True).dt.tz_convert(None)

            # Force specific types
            type_conversions = {
                "id": "string",
                "employee_id": "string",
                "branch": "string",
                "supplier": "string",
                "avg_time": "float64",
                "on_time": "boolean",
                "in_full": "boolean",
                "number_of_items": "Int32",
                "ft_items": "Int32",
                "total_price": "float64",
                "total_activities": "Int32",
                "rework_activities": "Int32",
                "automatic_activities": "Int32"
            }
            for col, dtype in type_conversions.items():
                if col in chunk_df.columns:
                    chunk_df[col] = chunk_df[col].astype(dtype)

            # Register chunk
            self.conn.register("temp_cases", chunk_df)

            if first_chunk:
                # Create the table from first chunk
                self.conn.execute("""
                    CREATE TABLE cases AS
                    SELECT 
                        id,
                        order_date,
                        employee_id,
                        branch,
                        supplier,
                        avg_time,
                        estimated_delivery,
                        delivery,
                        on_time,
                        in_full,
                        number_of_items,
                        ft_items,
                        total_price,
                        total_activities,
                        rework_activities,
                        automatic_activities
                    FROM temp_cases
                """)
                first_chunk = False
            else:
                # Insert next chunks
                self.conn.execute("""
                    INSERT INTO cases
                    SELECT 
                        id,
                        order_date,
                        employee_id,
                        branch,
                        supplier,
                        avg_time,
                        estimated_delivery,
                        delivery,
                        on_time,
                        in_full,
                        number_of_items,
                        ft_items,
                        total_price,
                        total_activities,
                        rework_activities,
                        automatic_activities
                    FROM temp_cases
                """)

            # Drop temp view after using it
            self.conn.execute("DROP VIEW temp_cases")

        row_count = self.conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        print(f"Loaded {row_count} cases")


    def process_activities(self):
        """Process Activity.json in chunks and load into DuckDB with proper schema."""
        print("Loading activities data in chunks...")
        self.conn.execute("DROP TABLE IF EXISTS activities")
        
        first_chunk = True
        for chunk_df in self.load_json_in_chunks("Activity.json"):
            if chunk_df.empty:
                continue

            # Flatten the "case" field
            if "case" in chunk_df.columns:
                case_df = pd.json_normalize(chunk_df["case"])
                case_df.columns = [f"case_{col}" for col in case_df.columns]
                chunk_df = pd.concat([chunk_df.drop(columns=["case"]), case_df], axis=1)

            # Convert timestamps
            if "timestamp" in chunk_df.columns:
                chunk_df["timestamp"] = pd.to_datetime(chunk_df["timestamp"], utc=True).dt.tz_convert(None)

            if "case_order_date" in chunk_df.columns:
                chunk_df["case_order_date"] = pd.to_datetime(chunk_df["case_order_date"], utc=True).dt.tz_convert(None)
            if "case_estimated_delivery" in chunk_df.columns:
                chunk_df["case_estimated_delivery"] = pd.to_datetime(chunk_df["case_estimated_delivery"], utc=True).dt.tz_convert(None)
            if "case_delivery" in chunk_df.columns:
                chunk_df["case_delivery"] = pd.to_datetime(chunk_df["case_delivery"], utc=True).dt.tz_convert(None)

            # Define expected data types
            type_conversions = {
                "id": "INTEGER",
                "timestamp": "TIMESTAMP",
                "name": "VARCHAR",
                "tpt": "DOUBLE",
                "user": "VARCHAR",
                "user_type": "VARCHAR",
                "automatic": "BOOLEAN",
                "rework": "BOOLEAN",
                "case_index": "INTEGER",
                # Flattened case fields
                "case_id": "VARCHAR",
                "case_order_date": "TIMESTAMP",
                "case_employee_id": "VARCHAR",
                "case_branch": "VARCHAR",
                "case_supplier": "VARCHAR",
                "case_avg_time": "DOUBLE",
                "case_estimated_delivery": "TIMESTAMP",
                "case_delivery": "TIMESTAMP",
                "case_on_time": "BOOLEAN",
                "case_in_full": "BOOLEAN",
                "case_number_of_items": "INTEGER",
                "case_ft_items": "INTEGER",
                "case_total_price": "DOUBLE"
            }

            # Register and insert chunk
            self.conn.register("temp_activities", chunk_df)

            if first_chunk:
                # Build CREATE TABLE with correct types
                columns_def = ",\n".join(f"{col} {dtype}" for col, dtype in type_conversions.items())
                create_table_sql = f"CREATE TABLE activities ({columns_def})"
                self.conn.execute(create_table_sql)

                # Insert first chunk
                self.conn.execute("INSERT INTO activities SELECT * FROM temp_activities")
                first_chunk = False
            else:
                self.conn.execute("INSERT INTO activities SELECT * FROM temp_activities")

            # Drop temporary view
            self.conn.execute("DROP VIEW temp_activities")

        row_count = self.conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]
        print(f"Loaded {row_count} activities")


    def process_variants(self):
        """Process Variant.json in chunks and load into DuckDB with proper schema."""
        print("Loading variants data in chunks...")
        self.conn.execute("DROP TABLE IF EXISTS variants")

        first_chunk = True
        for chunk_df in self.load_json_in_chunks("Variant.json"):
            if chunk_df.empty:
                continue

            # Convert string representations of lists to real Python lists
            list_cols = ["activities", "cases"]
            for col in list_cols:
                if col in chunk_df.columns:
                    chunk_df[col] = chunk_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

            # Register temp view
            self.conn.register("temp_variants", chunk_df)

            if first_chunk:
                # ✅ Create the table directly from the DataFrame with correct native types
                self.conn.execute("CREATE TABLE variants AS SELECT * FROM temp_variants")
                first_chunk = False
            else:
                # ✅ Insert into existing table
                self.conn.execute("INSERT INTO variants SELECT * FROM temp_variants")

            # Drop temp view
            self.conn.execute("DROP VIEW temp_variants")

        row_count = self.conn.execute("SELECT COUNT(*) FROM variants").fetchone()[0]
        print(f"Loaded {row_count} variants")

    def process_grouped(self):
            print("Loading grouped data in chunks...")
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

            row_count = self.conn.execute("SELECT COUNT(*) FROM grouped").fetchone()[0]
            print(f"Loaded {row_count} grouped entries")

    def process_invoices(self):
        print("Loading invoices data in chunks...")
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

        row_count = self.conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
        print(f"Loaded {row_count} invoices")




    def inspect_database(self):
        """Inspect the database structure and sample data"""
        print("\nDatabase Inspection:")
        tables = self.conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
        """).fetchall()

        for (table,) in tables:
            print(f"\n=== {table.upper()} ===")
            
            # Show structure
            print("\nStructure:")
            structure = self.conn.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table}'
            """).df()
            print(structure.to_string(index=False))
            
            # Show sample
            print("\nSample Data (5 rows):")
            sample = self.conn.execute(f"SELECT * FROM {table} LIMIT 5").df()
            print(sample.to_string(index=False))

    def run(self):
        self.process_cases()
        self.process_activities()
        self.process_variants()
        self.process_grouped()
        self.process_invoices()
        self.inspect_database()


# Usage
if __name__ == "__main__":
    loader = CargaDeArchivos()
    loader.run()
