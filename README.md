# EEG-CLIP: Contrastive Learning for EEG Classification

A PyTorch implementation of EEG-CLIP for motor imagery classification using contrastive learning. This project implements two variants of CLIP-style training for EEG signals:

1. **4-Class Fixed Templates**: Pre-computed text embeddings for 4 fixed class descriptions
2. **Batch-wise Contrastive Learning**: Dynamic caption generation with symmetric contrastive loss

## Features

- ✅ **Two Training Approaches**:
  - 4-class fixed templates (efficient, interpretable)
  - Batch-wise contrastive learning (better embeddings, more flexible)

- ✅ **Flexible Validation Strategies**:
  - Train-test split
  - K-fold cross-validation (subject-wise or trial-wise)

- ✅ **Comprehensive Evaluation**:
  - Accuracy, precision, recall, F1 scores
  - Confusion matrices
  - t-SNE embedding visualizations
  - Per-class metrics

- ✅ **Modular Architecture**:
  - ATCNet encoder with direct embedding output
  - Frozen pre-trained text encoder
  - Configurable via YAML files

- ✅ **Production Ready**:
  - PyTorch Lightning integration
  - Model checkpointing and early stopping
  - TensorBoard and W&B logging support
  - Reproducible experiments with seed control

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd EEG-Clip

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
EEG-Clip/
├── config/
│   ├── base_config.yaml           # Base configuration
│   └── experiments/
│       ├── 4class_fixed.yaml      # 4-class fixed experiment
│       ├── batch_contrastive.yaml # Batch contrastive experiment
│       └── kfold_4class.yaml      # K-fold experiment
├── src/
│   ├── models/
│   │   ├── atcnet.py              # ATCNet encoder
│   │   ├── text_encoder.py        # Frozen text encoder
│   │   ├── eeg_clip_4class.py     # 4-class fixed model
│   │   └── eeg_clip_batch.py      # Batch contrastive model
│   ├── data/
│   │   ├── dataset.py             # EEG dataset
│   │   └── caption_generator.py   # Caption generation
│   ├── training/
│   │   ├── trainer_4class.py      # 4-class trainer
│   │   └── trainer_batch.py       # Batch contrastive trainer
│   ├── evaluation/
│   │   └── evaluator.py           # Evaluation module
│   └── utils/
│       ├── visualization.py       # Plotting utilities
│       ├── config.py              # Config utilities
│       └── data_utils.py          # Data split utilities
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Quick Start

### 1. Configure Your Data

Edit `config/base_config.yaml` to point to your data:

```yaml
data:
  data_prefix: "/path/to/your/data/"
  subjects:
    subject1:
      - "Subject 1/0/0-raw.fif"
      - "Subject 1/1/1-raw.fif"
    subject2:
      - "Subject 2/0/0-raw.fif"
      - "Subject 2/1/1-raw.fif"
    # ... more subjects
```

### 2. Train a Model

#### Option A: 4-Class Fixed Templates (Recommended for beginners)

```bash
# Train with default settings
python train.py --config config/experiments/4class_fixed.yaml
```

#### Option B: Batch-wise Contrastive Learning

```bash
# Train with batch contrastive loss
python train.py --config config/experiments/batch_contrastive.yaml
```

#### Option C: K-Fold Cross-Validation

```bash
# Run all folds
python train.py --config config/experiments/kfold_4class.yaml --all-folds

# Or run a specific fold
python train.py --config config/experiments/kfold_4class.yaml --fold 0
```

### 3. Evaluate a Trained Model

```bash
# Evaluate on test set
python evaluate.py \
    --checkpoint checkpoints/eeg_clip_4class_fixed/epoch=XX-val_acc=X.XXXX.ckpt \
    --config config/experiments/4class_fixed.yaml \
    --plot-embeddings
```

## Configuration

### Base Configuration (`config/base_config.yaml`)

Key configuration sections:

```yaml
# Model settings
model:
  type: "4class_fixed"  # or "batch_contrastive"
  num_electrodes: 50
  chunk_size: 1001
  embedding_dim: 384
  temperature: 0.07
  text_model: "all-MiniLM-L6-v2"

# Training settings
training:
  max_epochs: 50
  batch_size: 32
  learning_rate: 3.0e-4
  weight_decay: 1.0e-4

# Validation strategy
validation:
  strategy: "train_test_split"  # or "k_fold"
  train_test_split:
    test_subjects: ["subject1"]
    val_split: 0.2
  k_fold:
    n_folds: 5
    fold_strategy: "subject_wise"  # or "trial_wise"

# Caption templates
captions:
  template_style: "motor_imagery"
  # Options: "simple", "action_focused", "motor_imagery", "neural", "descriptive", "varied"
```

### Creating Custom Experiments

1. Copy an existing experiment config from `config/experiments/`
2. Modify the settings you want to change
3. Run training with your custom config:

```bash
python train.py --config config/experiments/my_experiment.yaml
```

## Model Architectures

### 4-Class Fixed Templates

```
ATCNet → [Direct Embedding] → 384-dim → Compare with 4 fixed text embeddings
                                           ↓
                                    Cross-Entropy Loss
```

**Advantages:**
- ✅ Efficient (text embeddings computed once)
- ✅ Interpretable (fixed class descriptions)
- ✅ Fast training
- ✅ Good for exactly 4 classes

### Batch-wise Contrastive Learning

```
EEG → ATCNet → 384-dim ─┐
                        ├─→ Similarity Matrix → Symmetric Contrastive Loss
Text → Frozen Encoder →─┘
```

**Advantages:**
- ✅ Better embeddings (harder task)
- ✅ Caption variation (multiple templates)
- ✅ Scalable to more classes
- ✅ Retrieval capability

## Results

### Expected Performance

| Metric | 4-Class Fixed | Batch Contrastive |
|--------|--------------|-------------------|
| **Validation Accuracy** | 75-85% | 70-80% |
| **Training Speed** | Faster | Slower |
| **Embedding Quality** | Good | Better |
| **Memory Usage** | Lower | Higher |

### Example Output

```
================================================================================
EVALUATION RESULTS
================================================================================

Overall Metrics:
  Accuracy:  78.33%
  Precision: 78.45%
  Recall:    78.33%
  F1 Score:  78.21%

Per-Class Metrics:

  left hand:
    Precision: 82.50%
    Recall:    75.45%
    F1 Score:  78.82%

  right hand:
    Precision: 76.92%
    Recall:    80.00%
    F1 Score:  78.43%

  foot:
    Precision: 77.27%
    Recall:    78.18%
    F1 Score:  77.72%

  rest:
    Precision: 77.14%
    Recall:    80.00%
    F1 Score:  78.55%
================================================================================
```

## Customization

### Adding New Caption Templates

Edit `src/data/caption_generator.py`:

```python
TEMPLATE_STYLES = {
    'my_custom_style': [
        "custom description for class 0",
        "custom description for class 1",
        "custom description for class 2",
        "custom description for class 3"
    ]
}
```

Then use it in your config:

```yaml
captions:
  template_style: "my_custom_style"
```

### Modifying ATCNet Architecture

Edit the `atcnet` section in your config:

```yaml
model:
  atcnet:
    num_windows: 3
    conv_pool_size: 7
    F1: 16
    D: 2
    tcn_kernel_size: 4
    tcn_depth: 2
```

### Using Different Text Encoders

Change the `text_model` parameter:

```yaml
model:
  text_model: "all-mpnet-base-v2"  # 768-dim embeddings
```

Available models:
- `all-MiniLM-L6-v2` (384-dim, fast)
- `all-mpnet-base-v2` (768-dim, better quality)

## Advanced Usage

### K-Fold Cross-Validation

Run all folds and get mean ± std results:

```bash
python train.py --config config/experiments/kfold_4class.yaml --all-folds
```

### Custom Data Splits

Specify custom train/validation/test splits in config:

```yaml
validation:
  strategy: "train_test_split"
  train_test_split:
    test_subjects: ["subject1", "subject2"]
    val_subjects: ["subject3"]
```

### Logging with Weights & Biases

Enable W&B in config:

```yaml
logging:
  wandb:
    enabled: true
    project: "eeg-clip"
    entity: "your-username"
```

### GPU Training

Specify device in config:

```yaml
experiment:
  device: "cuda"  # or "cpu" or "auto"
```

## Troubleshooting

### Out of Memory Errors

Reduce batch size in config:

```yaml
training:
  batch_size: 16  # or 8
```

### Slow Training

- Use GPU if available
- Reduce `num_workers` if CPU bottleneck
- Use smaller text encoder (`all-MiniLM-L6-v2`)

### Poor Performance

- Increase `max_epochs`
- Try different `learning_rate` (1e-4 to 5e-4)
- Experiment with caption templates
- Check data quality and preprocessing

## Citation

If you use this code in your research, please cite:

```bibtex
@article{eeg-clip-2024,
  title={EEG-CLIP: Contrastive Learning for EEG Motor Imagery Classification},
  author={Your Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- ATCNet architecture based on [ATCNet paper](https://ieeexplore.ieee.org/document/9852687)
- CLIP framework inspired by OpenAI's CLIP
- Text encoder from [sentence-transformers](https://www.sbert.net/)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub.