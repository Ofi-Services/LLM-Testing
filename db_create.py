import os
import pandas as pd
import json
import duckdb

class CargaDeArchivos:
    def __init__(self):
        self.carpeta_cases = "./Data/cases"
        self.carpeta_activities = "./Data/activity"
        self.case_df = None
        self.activities_df = None
        self.conn = duckdb.connect(":memory:")  # Base de datos en memoria
        self.chunksize = 1000  # No es obligatorio en DuckDB, pero lo mantenemos

    def carga_cases(self):
        """Carga todos los archivos JSON de la carpeta de cases en un DataFrame y convierte fechas a TIMESTAMP."""
        try:
            case_list = []
            for archivo in os.listdir(self.carpeta_cases):
                if archivo.endswith(".json"):
                    ruta_completa = os.path.join(self.carpeta_cases, archivo)
                    with open(ruta_completa, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        df = pd.DataFrame(data.get("results", []))

                    # Renombrar la columna mal escrita "brocker" -> "broker"
                    df.rename(columns={"brocker": "broker"}, inplace=True)

                    # Convertir fechas a TIMESTAMP sin timezone
                    date_cols = ["insurance_creation", "insurance_start", "insurance_end"]
                    for col in date_cols:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)

                    case_list.append(df)

            self.case_df = pd.concat(case_list, ignore_index=True) if case_list else pd.DataFrame()
            return self.case_df
        except Exception as e:
            print(f"Error al cargar los datos de los casos: {e}")

    def carga_activities(self):
        """Carga todos los archivos JSON de la carpeta de activities en un DataFrame y convierte fechas a TIMESTAMP."""
        try:
            activity_list = []
            for archivo in os.listdir(self.carpeta_activities):
                if archivo.endswith(".json"):
                    ruta_completa = os.path.join(self.carpeta_activities, archivo)
                    with open(ruta_completa, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        df = pd.DataFrame(data.get("results", []))

                    # Convertir fechas a TIMESTAMP sin timezone
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)

                    activity_list.append(df)

            self.activities_df = pd.concat(activity_list, ignore_index=True) if activity_list else pd.DataFrame()
            return self.activities_df
        except Exception as e:
            print(f"Error al cargar los datos de las actividades: {e}")

    def dataBase(self, activity, case):
        """Carga los DataFrames en DuckDB asegurando que los timestamps estén correctamente formateados."""
        try:
            if activity is not None and not activity.empty:
                self.conn.register("activity", activity)
                self.conn.execute("CREATE TABLE activity AS SELECT * FROM activity")

            if case is not None and not case.empty:
                self.conn.register("cases", case)

                # Asegurar que "broker" se usa correctamente en la tabla "cases"
                self.conn.execute("""
                    CREATE TABLE cases AS 
                    SELECT 
                        id, insurance, avg_time, type, branch, ramo, 
                        broker, state, client, creator, value, approved, 
                        insurance_creation, insurance_start, insurance_end 
                    FROM cases
                """)

        except Exception as e:
            print(f"Error al cargar datos en DuckDB: {e}")

    def consultar_db(self, query):
        """Ejecuta una consulta SQL en la base de datos en memoria."""
        try:
            return self.conn.execute(query).df()
        except Exception as e:
            print(f"Error en la consulta SQL: {e}")

    def run_carga(self):
        """Ejecuta la carga de archivos y los inserta en DuckDB."""
        cases = self.carga_cases()
        activities = self.carga_activities()

        if cases is not None and not cases.empty and activities is not None and not activities.empty:
            self.dataBase(activities, cases)
        else:
            print("No se pudieron cargar los datos correctamente.")
