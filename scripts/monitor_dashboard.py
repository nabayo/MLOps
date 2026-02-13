"""
MLflow Monitoring Dashboard

Usage:
    docker compose run --rm mlflow-server python scripts/monitor_dashboard.py

    docker compose run --rm mlflow-server python scripts/monitor_dashboard.py --export-all
"""

from typing import Any
from datetime import datetime

import os
import sys
import psycopg2

import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient


# --- Colors for TUI ---
class Colors:
    """
    Terminal colors for the dashboard.
    """

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text: str) -> None:
    """
    Print a header for the dashboard.
    """
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {text} ==={Colors.ENDC}")


def print_sub(text: str) -> None:
    """
    Print a sub-header for the dashboard.
    """
    print(f"\n{Colors.CYAN}--- {text} ---{Colors.ENDC}")


def print_error(text: str) -> None:
    """
    Print an error message.
    """
    print(f"{Colors.FAIL}Error: {text}{Colors.ENDC}")


def print_success(text: str) -> None:
    """
    Print a success message.
    """
    print(f"{Colors.GREEN}{text}{Colors.ENDC}")


def input_prompt(text: str) -> str:
    """
    Print a prompt and return the user's input.
    """
    return input(f"{Colors.BLUE}{text}{Colors.ENDC} ")


# --- Main Application ---
class Dashboard:
    """
    Main application class for the MLflow Monitoring Dashboard.
    """

    def __init__(self) -> None:
        """
        Initialize the dashboard.
        """
        self.tracking_uri: str = os.environ.get(
            "MLFLOW_TRACKING_URI", "http://mlflow:5000"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client: MlflowClient = MlflowClient()
        self.conn: psycopg2.connect | None = None
        self.connect_db()

    def connect_db(self) -> None:
        """
        Connect to the database.
        """

        try:
            # Attempt to connect to postgres using standard env vars if available,
            # or fallback to service name 'postgres' which works in docker compose network
            db_user: str = os.environ.get("POSTGRES_USER", "postgres")
            db_password: str = os.environ.get(
                "POSTGRES_PASSWORD", "postgres"
            )  # Default might be different
            db_name: str = os.environ.get("POSTGRES_DB", "mlflow")
            db_host: str = os.environ.get("POSTGRES_HOST", "postgres")

            self.conn = psycopg2.connect(
                host=db_host, database=db_name, user=db_user, password=db_password
            )
            print_success("Connected to Metadata Store (Postgres)")
        except Exception as e:  # pylint: disable=broad-except
            print_error(f"Could not connect to database: {e}")
            self.conn = None

    def run(self) -> None:
        """
        Run the dashboard.
        """

        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print_header("MLOps Monitoring Dashboard")
            print(f"Tracking URI: {self.tracking_uri}")
            print(f"Database: {'Connected' if self.conn else 'Disconnected'}")
            print("\n1. Experiment Browser")
            print("2. Registry Explorer")
            print("3. Metadata Store Explorer (SQL)")
            print("0. Exit")

            choice = input_prompt("Select option:")

            if choice == "1":
                self.menu_experiments()
            elif choice == "2":
                self.menu_registry()
            elif choice == "3":
                self.menu_metadata()
            elif choice == "0":
                if self.conn:
                    self.conn.close()
                sys.exit(0)
            else:
                input("Invalid option. Press Enter.")

    # --- Experiment Browser ---
    def menu_experiments(self) -> None:
        """
        Menu for experiments.
        """

        while True:
            print_sub("Experiment Browser")
            experiments = self.client.search_experiments()
            data = []
            for exp in experiments:
                data.append([exp.experiment_id, exp.name, exp.lifecycle_stage])

            print(
                pd.DataFrame(data, columns=["ID", "Name", "Stage"]).to_string(
                    index=False
                )
            )
            print("\nActions:")
            print("  [ID] to view runs in experiment")
            print("  'del [ID]' to delete experiment")
            print("  'b' to go back")

            choice = input_prompt("Action:")
            if choice == "b":
                break
            if choice.startswith("del "):
                exp_id = choice.split(" ")[1]
                confirm = input_prompt(
                    f"Are you sure you want to delete experiment {exp_id}? (y/n)"
                )
                if confirm.lower() == "y":
                    try:
                        self.client.delete_experiment(exp_id)
                        print_success(f"Deleted experiment {exp_id}")
                    except Exception as e:  # pylint: disable=broad-except
                        print_error(str(e))
                    input("Press Enter...")
            else:
                # Assume ID
                found = False
                for exp in experiments:
                    if exp.experiment_id == choice:
                        self.view_runs(choice)
                        found = True
                        break
                if not found:
                    input("Experiment not found. Press Enter.")

    def view_runs(self, experiment_id: str) -> None:
        """
        View runs in an experiment.
        """

        while True:
            print_sub(f"Runs in Experiment {experiment_id}")
            runs = self.client.search_runs(
                experiment_id, order_by=["attribute.start_time DESC"]
            )

            if not runs:
                print("No runs found.")
            else:
                data = []
                for run in runs:
                    start = datetime.fromtimestamp(run.info.start_time / 1000).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    status = run.info.status
                    run_id = run.info.run_id
                    data.append([run_id, start, status])
                print(
                    pd.DataFrame(data, columns=["Run ID", "Start Time", "Status"])
                    .head(20)
                    .to_string(index=False)
                )

            print("\nActions:")
            print("  [Run ID] to view details/artifacts")
            print("  'del [Run ID]' to delete run")
            print("  'reg [Run ID]' to force register model from this run")
            print("  'b' to go back")

            choice = input_prompt("Action:")
            if choice == "b":
                break
            if choice.startswith("del "):
                run_id = choice.split(" ")[1]
                try:
                    self.client.delete_run(run_id)
                    print_success(f"Deleted run {run_id}")
                except Exception as e:  # pylint: disable=broad-except
                    print_error(str(e))
                input("Press Enter...")
            elif choice.startswith("reg "):
                run_id = choice.split(" ")[1]
                self.force_register_model(run_id)
                input("Press Enter...")
            else:
                # Check if run exists in the list we just fetched
                target_run = next((r for r in runs if r.info.run_id == choice), None)
                if target_run:
                    self.view_run_details(target_run)
                else:
                    input("Run not found. Press Enter to continue...")

    def view_run_details(self, run: Any) -> None:
        """
        View details of a run.
        """

        print_sub(f"Run Details: {run.info.run_id}")
        print(f"Status: {run.info.status}")
        print(f"Artifact URI: {run.info.artifact_uri}")

        print("\nParams:")
        for k, v in run.data.params.items():
            print(f"  {k}: {v}")

        print("\nMetrics:")
        for k, v in run.data.metrics.items():
            print(f"  {k}: {v}")

        print("\nArtifacts (Top Level):")
        try:
            artifacts = self.client.list_artifacts(run.info.run_id)
            for art in artifacts:
                print(f"  - {art.path} ({art.file_size} bytes)")
        except Exception as e:  # pylint: disable=broad-except
            print_error(f"Could not list artifacts: {e}")

        input("\nPress Enter to go back...")

    def force_register_model(self, run_id):
        """
        Force register a model from a run.
        """
        print_sub(f"Force Register Model from Run {run_id}")
        model_name = input_prompt("Enter Model Name (default: YOLOv11-Finger-Counter):")
        if not model_name:
            model_name = "YOLOv11-Finger-Counter"

        try:
            self.client.create_registered_model(model_name)
            print_success(f"Created/Ensured Registered Model: {model_name}")
        except Exception:  # pylint: disable=broad-except
            pass  # Already exists

        # Default path for this project seems to be 'weights/model.pt' based on force_register.py
        artifact_path = input_prompt("Enter Artifact Path (default: weights/model.pt):")
        if not artifact_path:
            artifact_path = "weights/model.pt"

        source = f"runs:/{run_id}/{artifact_path}"
        print(f"Source: {source}")

        try:
            mv = self.client.create_model_version(
                name=model_name, source=source, run_id=run_id
            )
            print_success(f"Successfully registered version {mv.version}!")
        except Exception as e:  # pylint: disable=broad-except
            print_error(f"Registration failed: {e}")

    # --- Registry Explorer ---
    def menu_registry(self):
        while True:
            print_sub("Registry Explorer")
            models = self.client.search_registered_models()

            if not models:
                print("No registered models found.")
            else:
                data = []
                for m in models:
                    # Get latest versions
                    latest = ", ".join(
                        [f"v{v.version}({v.current_stage})" for v in m.latest_versions]
                    )
                    data.append([m.name, latest])
                print(
                    pd.DataFrame(
                        data, columns=["Model Name", "Latest Versions"]
                    ).to_string(index=False)
                )

            print("\nActions:")
            print("  'del [Name]' to delete model")
            print("  'b' to go back")

            choice = input_prompt("Action:")
            if choice == "b":
                break
            if choice.startswith("del "):
                name = choice.split(" ", 1)[1]
                confirm = input_prompt(
                    f"DELETE model {name}? This cannot be undone. (y/n)"
                )
                if confirm.lower() == "y":
                    try:
                        self.client.delete_registered_model(name)
                        print_success(f"Deleted {name}")
                    except Exception as e:  # pylint: disable=broad-except
                        print_error(f"Failed: {e}")
                    input("Press Enter...")
            else:
                input("Invalid action. Press Enter.")

    def menu_metadata(self):
        """
        Menu for metadata store exploration.
        """
        if not self.conn:
            print_error("Database connection not available.")
            print(
                "Ensure POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, "
                "POSTGRES_HOST are set correctly."
            )
            input("Press Enter...")
            return

        while True:
            print_sub("Metadata Store (Postgres) Explorer")
            print("1. List Tables")
            print("2. Run Custom SQL Query")
            print("3. Show recent experiment modifications (SQL)")
            print("b. Back")

            choice = input_prompt("Option:")

            if choice == "b":
                break

            cursor = self.conn.cursor()

            if choice == "1":
                try:
                    cursor.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public';"
                    )
                    tables = cursor.fetchall()
                    print("\nTables:")
                    for t in tables:
                        print(f"  - {t[0]}")
                except Exception as e:  # pylint: disable=broad-except
                    print_error(str(e))
                input("\nPress Enter...")

            elif choice == "2":
                self._run_custom_query(cursor)

            elif choice == "3":
                # Example useful query
                try:
                    cursor.execute(
                        "SELECT experiment_id, name, lifecycle_stage "
                        "FROM experiments LIMIT 10;"
                    )
                    rows = cursor.fetchall()
                    print("\nExperiments (Raw SQL):")
                    for r in rows:
                        print(r)
                except Exception as e:  # pylint: disable=broad-except
                    print_error(str(e))
                input("\nPress Enter...")

            cursor.close()

    def _run_custom_query(self, cursor: Any) -> None:
        """Run a custom SQL query."""
        query = input_prompt("SQL Query >")
        if query:
            try:
                cursor.execute(query)
                if cursor.description:
                    cols = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    if rows:
                        print(pd.DataFrame(rows, columns=cols).to_string())
                    else:
                        print("No results.")
                else:
                    self.conn.commit()
                    print_success("Query executed.")
            except Exception as e:  # pylint: disable=broad-except
                self.conn.rollback()
                print_error(str(e))
            input("\nPress Enter...")


if __name__ == "__main__":
    try:
        app = Dashboard()
        app.run()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
