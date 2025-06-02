# Time Series Pump Prediction Model v2

A PyTorch-based time series forecasting model for predicting cryptocurrency token price movements using transformer architecture.

## Overview

This project implements a deep learning model that predicts token price movements using historical price, volume, liquidity, buy/sell data, and market cap features. The model employs a transformer encoder architecture with positional encoding for time series forecasting.

## Features

- **Transformer-based Architecture**: Uses PyTorch transformer encoder layers for sequence modeling
- **Multi-token Training**: Supports training on multiple synthetic token datasets
- **Multiple Data Split Strategies**: Time-based, token-based, and mixed splitting approaches
- **Hyperparameter Optimization**: Integrated Optuna optimization for automated hyperparameter tuning
- **MLflow Integration**: Experiment tracking and model versioning
- **Comprehensive Visualization**: Feature correlation analysis and prediction plotting
- **Flexible Prediction**: Short-term and extended forecasting capabilities

## Model Architecture

- **Input Embedding**: Conv1D or Linear embedding (6 features → embed_size)
- **Positional Encoding**: Sinusoidal positional encoding for sequence awareness
- **Transformer Encoder**: Multi-head attention mechanism for pattern recognition
- **Regression Head**: Multi-layer feedforward network for price prediction

### Features Used
1. **Price**: Token price (primary prediction target)
2. **Volume**: Trading volume
3. **Liquidity**: Available liquidity
4. **Buys**: Buy pressure
5. **Sells**: Sell pressure
6. **Market Cap**: Calculated market capitalization

## Installation

### Dependencies
```bash
pip install torch numpy matplotlib seaborn h5py mlflow optuna
```

### Required Libraries
- PyTorch (CUDA support recommended)
- NumPy
- Matplotlib
- Seaborn
- H5py
- MLflow
- Optuna

## Usage

### Basic Training
```bash
python pump_trainer.py --epochs 30 --batch-size 4 --learning-rate 2.2e-6
```

### Training with Custom Parameters
```bash
python pump_trainer.py \
    --seq-length 500 \
    --embed-size 64 \
    --nhead 4 \
    --dim-feedforward 256 \
    --num-tokens 10 \
    --timesteps 4000 \
    --epochs 50 \
    --save-model model_v2.pth
```

### Testing a Trained Model
```bash
python pump_trainer.py \
    --test-mode \
    --load-model model_v2.pth \
    --test-token-id 0 \
    --forecast 1000 \
    --forecast-extended 3000
```

### Testing Multiple Tokens
```bash
python pump_trainer.py \
    --test-mode \
    --load-model model_v2.pth \
    --test-token-range 0-5 \
    --show-data-splits
```

## Data Split Strategies

### Time-based Split (`--split-strategy time`)
- Splits each token's time series into train/validation/test chronologically
- Prevents data leakage by using different time periods
- Default: 40% test, 20% validation, 40% training

### Token-based Split (`--split-strategy token`)
- Uses different tokens entirely for training/validation/testing
- Tests model generalization to completely unseen tokens
- Configurable with `--train-token-ratio`

### Mixed Split (`--split-strategy mixed`)
- Combines token-based and time-based approaches
- Different tokens for train/val, time-based split for test
- Evaluates both temporal and cross-token generalization

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seq-length` | int | 500 | Input sequence length |
| `--embed-size` | int | 64 | Embedding dimension |
| `--nhead` | int | 4 | Number of attention heads |
| `--dim-feedforward` | int | 256 | Feedforward network size |
| `--dropout` | float | 0.0 | Dropout rate |
| `--epochs` | int | 30 | Training epochs |
| `--batch-size` | int | 1 | Batch size |
| `--learning-rate` | float | 2.2e-6 | Learning rate |
| `--num-tokens` | int | 5 | Number of synthetic tokens |
| `--timesteps` | int | 4000 | Timesteps per token |
| `--save-model` | str | None | Model save path |
| `--load-model` | str | None | Model load path |
| `--test-mode` | flag | False | Run testing only |
| `--forecast` | int | 1000 | Short-term forecast steps |
| `--forecast-extended` | int | 3000 | Extended forecast steps |
| `--split-strategy` | str | time | Data splitting strategy |
| `--show-data-splits` | flag | False | Visualize data splits |

## Data Management

The project uses `TokenDataManager` for efficient data handling:

- **HDF5 Storage**: Compressed storage for large datasets
- **Metadata Tracking**: JSON-based token address to ID mapping
- **Multi-token Support**: Seamless handling of multiple token datasets

### Data Structure
```
training-data/
├── sequences/
│   ├── token_0.h5
│   ├── token_1.h5
│   └── ...
├── token_metadata.json
├── normalization_params.npy
└── split_info.json
```

## Output Files

### Models
- `model_v2.pth`: Trained model checkpoint

### Visualizations
- `img/token_X/features.png`: Feature time series plots
- `img/token_X/correlations.png`: Feature correlation heatmaps
- `predictions/prediction_short.png`: Short-term predictions
- `predictions/prediction_extended.png`: Extended predictions

### Data Files
- Normalized token sequences in HDF5 format
- Token metadata and normalization parameters
- Data split information for reproducibility

## MLflow Tracking

The project integrates with MLflow for experiment management:

```bash
# View experiments in MLflow UI
mlflow ui
```

Tracked metrics include:
- Training loss
- Validation loss (for mixed strategy)
- Model hyperparameters
- Model artifacts

## Advanced Features

### Hyperparameter Optimization
The trainer includes Optuna integration for automated hyperparameter search across:
- Embedding size
- Number of attention heads
- Sequence length
- Feedforward dimensions
- Dropout rate
- Learning rate
- Batch size

### Synthetic Data Generation
Generates realistic token data with:
- Pump and dump patterns
- News event impacts
- Market sentiment fluctuations
- Volume spikes and clusters
- Liquidity withdrawals
- Trading bursts

## File Structure

```
time-series-pump-v2/
├── forecasting_model.py      # Transformer model architecture
├── pump_trainer.py          # Training and testing script
├── token_data_manager.py    # Data management utilities
├── model_v2.pth            # Trained model checkpoint
├── training-data*/         # Training datasets (multiple versions)
├── img/                    # Visualization outputs
└── predictions/            # Prediction outputs
```

## Performance Considerations

- **GPU Support**: Automatic CUDA detection and usage
- **Memory Efficiency**: HDF5 storage for large datasets
- **Batch Processing**: Configurable batch sizes
- **Model Checkpointing**: Interrupt-safe model saving

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for research and educational purposes. Please ensure compliance with relevant regulations when working with financial data.