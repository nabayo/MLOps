# Projet MLOps - Présentation

---

## Slide - Titre

**Plateforme MLOps**
**Détection de Comptage de Doigts**

YOLOv11 • MLflow • Docker • FastAPI

---

## Slide - Table des matières

1. Description et architecture du projet
2. Données & Dataset
3. Entraînement & Déploiement
4. Démo & Conclusion

---

## Slide 01 - Architecture Générale

**Plateforme MLOps complète** pour l'entraînement et le déploiement

- **Infrastructure**: Docker Compose (7 services)
- **Tracking**: MLflow + PostgreSQL + MinIO
- **Modèle**: YOLOv11/26 (modèles nano / small / medium)
- **API**: FastAPI + Dashboard Web

---

## Slide 02 - Stack Technique

**Technologies utilisées**

- Python 3.12 + Ultralytics YOLO
- MLflow 2.18.0 (tracking & registry)
- Docker + Docker Compose
- FastAPI + Nginx

---

## Slide 03 - Services Docker

**7 services orchestrés**

- MLflow (tracking experiments)
- PostgreSQL (métadonnées)
- MinIO (stockage S3 des artefacts)
- API Serving (inférence temps réel)

---

## Slide 04 - Dataset

**Source des données**

- **315 images** de comptage de doigts
- Mix: photos personnelles + Roboflow dataset
- **7 classes**: 0, 1, 2, 3, 4, 5, unknown
- Gestion via **Picsellia**

---

## Slide 05 - Préparation des Données

**Pipeline automatisé**

- Téléchargement depuis Picsellia API
- Conversion: Picsellia → format YOLO
- Split: 80% train / 10% val / 10% test
- Validation des annotations

---

## Slide 06 - Format YOLO

**Conversion intelligente**

- Parsing des annotations UUID-based
- Normalisation des bounding boxes
- Filtrage des annotations "ACCEPTED"
- Gestion des erreurs (30 images corrompues)

---

## Slide 07 - Pipeline d'Entraînement

**Training automatisé avec MLflow**

- Configuration YAML flexible
- 5 modèles YOLOv11 (n, s, m, l, x)
- Logging automatique (params + metrics)
- Model Registry intégré

---

## Slide 08 - MLflow Integration

**Tracking exhaustif**

- Hyperparamètres (epochs, batch, lr)
- Métriques (mAP, precision, recall)
- Artefacts (modèles, plots, configs)
- Versioning automatique

---

## Slide 09 - Entraînement

**Configuration**

- 100 epochs, batch size 16
- Optimizer: auto (AdamW)
- Device: CPU/CUDA
- Early stopping: patience 100

---

## Slide 10 - Évaluation

**Métriques calculées**

- mAP@0.5 et mAP@0.5:0.95
- Precision, Recall, F1 par classe
- Confusion matrix multi-classes
- Validation automatique post-training

---

## Slide 11 - API de Serving

**FastAPI pour l'inférence**

- Endpoint `/predict` (POST image)
- Chargement auto des modèles MLflow
- Preprocessing: face blurring
- Documentation: `/docs`

---

## Slide 12 - Preprocessing

**Protection de la vie privée**

- Détection de visages (Haar Cascades)
- Floutage gaussien automatique
- Configurable via variable ENV
- Appliqué avant inférence

---

## Slide 13 - Dashboard Web

**Interface utilisateur moderne**

- Upload drag-and-drop
- Sélection de modèle
- Visualisation bounding boxes
- Affichage confidence scores

---

## Slide 14 - Scripts Cross-Platform

**Automatisation complète**

- Bash (Linux/Mac)
- PowerShell (Windows)
- Batch (Windows legacy)
- Fonctionnalités identiques

---

## Slide 15 - Opérations Supportées

**Scripts disponibles**

- `train` - Entraînement de modèle
- `serve` - Démarrage API
- `export` - Backup MLflow
- `import` - Restauration données

---

## Slide 16 - Backup/Restore

**Innovation technique**

- Export complet MLflow → ZIP
- Experiments + Runs + Models
- Gestion des collisions (skip existing)
- Mode dry-run pour preview

---

## Slide 17 - Défis Techniques Résolus

**Problèmes majeurs**

- Format Picsellia non-documenté ✓
- Compatibilité Python 3.12 + MLflow ✓
- Dépendances OpenCV Docker ✓
- Cross-platform scripts ✓

---

## Slide 18 - Architecture Docker

**Infrastructure robuste**

- Health checks sur tous services
- Volumes persistants (PostgreSQL, MinIO)
- Network isolation
- Restart policies configurées

---

## Slide 19 - Démo

**Fonctionnalités clés**

1. Training pipeline automatisé
2. MLflow UI (tracking + registry)
3. API inference temps réel
4. Dashboard interactif

---

## Slide 20 - Résultats

**Performance**

- 315 images processées
- ~285 images valides
- 7 classes détectées
- Inférence: ~200-500ms (CPU)

---

## Slide 21 - Documentation

**8 documents créés**

- README principal
- Guide d'entraînement
- Guide Windows
- Documentation backup/restore

---

## Slide 22 - Améliorations Futures

**Pistes d'évolution**

- Support GPU (CUDA)
- Monitoring (Prometheus + Grafana)
- Pipeline CI/CD
- Optimisation modèle (quantization)

---

## Slide 23 - Achievements

**Réalisations clés**

✅ Pipeline end-to-end complet
✅ Multi-plateforme (Linux/Mac/Windows)
✅ Production-ready API
✅ Documentation exhaustive

---

## Slide 24 - Conclusion

**MLOps Platform opérationnelle**

- Infrastructure complète dockerisée
- Training automatisé avec tracking
- Serving API production-ready
- Backup/Restore fonctionnel

**→ Prêt pour la production**

---

## Slide 25 - Questions ?

**Ressources**

- **Code**: github.com/nabayo/MLOps
- **MLflow**: http://localhost:5000
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8080

---
