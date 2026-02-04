# Training Scripts

Helper scripts for running the MLOps pipeline.

## Usage

### Training with Docker (Recommended)

```bash
./scripts/train.sh
```

This will:
- Start MLflow services if not running
- Build training container
- Run training with evaluation
- Log everything to MLflow

### Custom Training Commands

```bash
# Basic training
docker-compose run --rm training python main.py train

# Training with evaluation
docker-compose run --rm training python main.py train --evaluate

# Evaluation only
docker-compose run --rm training python main.py eval --model /app/models/best.pt

# Custom config
docker-compose run --rm training python main.py train --training-config /app/configs/custom.yaml
```

### Quick Commands

```bash
# Just build the training image
docker-compose build training

# Run training in background
docker-compose up -d training

# View training logs
docker-compose logs -f training
```
