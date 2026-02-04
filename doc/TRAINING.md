# Quick Training Guide

## ✅ Run Training with Docker (3 simple steps)

### 1. Make sure services are running
```bash
docker-compose up -d
```

### 2. Run training
```bash
docker-compose run --rm training
```

That's it! The training will:
- ✅ Load dataset from Picsellia
- ✅ Prepare data (YOLO format, 80/10/10 splits)
- ✅ Train YOLOv11
- ✅ Log everything to MLflow
- ✅ Register model in Model Registry
- ✅ Evaluate on test set

### 3. View results
```bash
open http://localhost:5000
```

## 📝 Custom Training

### Different model architecture
Edit `configs/training_config.yaml`:
```yaml
model:
  architecture: "yolo11s"  # Try: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
```

Then run:
```bash
docker-compose run --rm training
```

### Custom training config
```bash
docker-compose run --rm training python main.py train --training-config /app/configs/custom.yaml
```

### Training only (no evaluation)
```bash
docker-compose run --rm training python main.py train
```

### Evaluation only
```bash
docker-compose run --rm training python main.py eval --model /app/experiments/run_*/weights/best.pt
```

## 🔍 Monitoring

### View training logs in real-time
```bash
docker-compose run --rm training  # In one terminal
# Training output will stream here
```

### Check MLflow UI
```bash
open http://localhost:5000
```
Watch experiments, metrics, and artifacts in real-time!

## 🐛 Troubleshooting

### "picsellia_token not found"
Make sure you have the `picsellia_token` file in the root directory:
```bash
echo "your_token_here" > picsellia_token
```

### Rebuild training image
```bash
docker-compose build training
```

### View previous training logs
```bash
docker-compose logs training
```

### Clear dataset cache
```bash
rm -rf dataset/
```

## 💡 Pro Tips

### Use the helper script
```bash
./scripts/train.sh
```
This automatically handles service startup and training!

### Train multiple models
```bash
# Edit config for yolo11n
docker-compose run --rm training

# Edit config for yolo11s
docker-compose run --rm training

# Edit config for yolo11m
docker-compose run --rm training

# Compare all in MLflow!
```

### Background training
```bash
docker-compose up -d training
docker-compose logs -f training
```

Stop with:
```bash
docker-compose down training
```
