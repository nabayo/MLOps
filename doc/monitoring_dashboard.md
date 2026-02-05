# MLOps Monitoring Dashboard

The **Monitoring Dashboard** is a terminal-based interface that allows you to inspect and manage your MLflow experiments, Model Registry, and Metadata Store directly from within the Docker environment.

## Features

- **Experiment Browser**: View all experiments and their runs. Delete experiments or runs.
- **Run Inspector**: View parameters, metrics, and artifact lists for specific runs.
- **Model Registry**: View registered models and versions. Delete models.
- **Force Registration**: Manually register a model from a specific Run ID (useful for recovery).
- **Metadata Explorer**: Execute raw SQL queries against the Postgres backend.

## How to Run

Ensure your Docker stack is running:
```bash
docker compose up -d
```

Then, use one of the helper scripts provided in the `scripts/` directory:

### Windows (PowerShell)
```powershell
./scripts/monitor.ps1
```

### Windows (CMD)
```cmd
scripts\monitor.bat
```

### Linux / Mac (Bash)
```bash
./scripts/monitor.sh
```

## Manual Launch

You can also launch it manually using `docker compose`:

```bash
docker compose exec -it training python scripts/monitor_dashboard.py
```
(You can also use the `export` service if `training` is not running/needed).

## Database Connection
The script automatically connects to the Postgres database using environment variables defined in `docker-compose.yml`.

## Troubleshooting
- **Connection Refused**: Ensure the `postgres` and `mlflow` services are healthy (`docker compose ps`).
- **Missing Dependencies**: The script requires `mlflow`, `pandas`, `psycopg2`. These are installed in the `training` and `export` containers.
