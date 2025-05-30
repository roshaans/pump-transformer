# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from forecasting_model import ForecastingModel
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader
from token_data_manager import TokenDataManager, MultiTokenDataset
import h5py
import seaborn as sns
import os
import argparse
import mlflow
import mlflow.pytorch

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a price prediction model')
mlflow.set_tracking_uri("https://public-tracking.mlflow-e00bw0ayfax5408rrr.backbone-e00t0zs5ecdj7bwqg0.msp.eu-north1.nebius.cloud")  # Set this to your desired tracking URI
mlflow.set_experiment("PumpFun Token Price Prediction")
parser.add_argument('--seq-length', type=int, default=500, help='Sequence length for the model')
parser.add_argument('--save-model', type=str, help='Path to save the trained model')
parser.add_argument('--load-model', type=str, help='Path to load a pre-trained model')
parser.add_argument('--save-training-data', action='store_true', help='Save the training data')
parser.add_argument('--data-dir', type=str, default='./training-data', help='Directory to save training data')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training')
parser.add_argument('--learning-rate', type=float, default=2.2e-6, help='Learning rate')
parser.add_argument('--forecast', type=int, default=1000, help='Number of steps to forecast')
parser.add_argument('--forecast-extended', type=int, default=3000, help='Number of steps for extended forecast')
parser.add_argument('--embed-size', type=int, default=64, help='Embedding size for the model')
parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
parser.add_argument('--dim-feedforward', type=int, default=256, help='Dimension of feedforward network')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--num-tokens', type=int, default=5, help='Number of token datasets to generate')
parser.add_argument('--timesteps', type=int, default=4000, help='Number of timesteps to generate per token')
parser.add_argument('--train-examples', type=int, default=None, help='Number of training examples to use (default: all available)')
parser.add_argument('--test-mode', action='store_true', help='Run in test mode only (no training)')
parser.add_argument('--test-token-id', type=int, default=None, help='Specific token ID to test on')
parser.add_argument('--test-token-range', type=str, default=None, help='Range of token IDs to test on (format: start-end, e.g., 0-5)')
parser.add_argument('--test-offset', type=int, default=None, help='Offset from start of data to begin testing (default: last seq_len points)')
parser.add_argument('--output-dir', type=str, default='./predictions', help='Directory to save prediction outputs')
parser.add_argument('--list-tokens', action='store_true', help='List all available tokens in the data directory')
parser.add_argument('--show-data-splits', action='store_true', help='Show training/validation/test data splits in prediction plots')
parser.add_argument('--split-strategy', type=str, default='time', choices=['time', 'token', 'mixed'], 
                    help='Strategy for splitting data: "time" (split each token), "token" (use different tokens), or "mixed" (token-based for train/val, time-based for test)')
parser.add_argument('--train-token-ratio', type=float, default=0.6, 
                    help='Proportion of tokens to use for training when using token-based split strategy')
parser.add_argument('--tokens-limit', type=int, default=None, help='Limit the number of existing tokens to use for training (default: None, generate new tokens)')
args = parser.parse_args()

# Define img_dir
script_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(script_dir, "img")

def generate_token_data(token_manager: TokenDataManager, token_address: str, timesteps: int = None, 
                        train_split=0.4, val_split=0.2, test_split=0.4):
    """
    Generate synthetic token data and save it using the token manager
    Args:
        token_manager: TokenDataManager instance to save the data
        token_address: Token contract address
        timesteps: Number of time intervals to generate
        train_split: Proportion of data to use for training
        val_split: Proportion of data to use for validation
        test_split: Proportion of data to use for testing
    Returns:
        tuple: (train_data, val_data, test_data, full_data) normalized numpy arrays
    """
    # Use the timesteps from args if not explicitly provided
    if timesteps is None:
        timesteps = args.timesteps
    
    # Ensure splits sum to 1
    total_split = train_split + val_split + test_split
    if total_split != 1.0:
        train_split = train_split / total_split
        val_split = val_split / total_split
        test_split = test_split / total_split
    """
    Generate synthetic token data and save it using the token manager
    Args:
        token_manager: TokenDataManager instance to save the data
        token_address: Token contract address
        timesteps: Number of time intervals to generate
    Returns:
        tuple: (train_data, val_data) normalized numpy arrays
    """
    t = np.linspace(0, 8*np.pi, timesteps)
    
    # Price: Combine multiple patterns with more realistic behavior
    trend = t/3  # Overall upward trend
    long_cycle = 15*np.sin(t/4)  # Long cycle
    medium_cycle = 8*np.sin(t)  # Medium cycle
    short_cycle = 4*np.sin(3*t)  # Short cycle
    
    # Initialize price array
    price = np.zeros(timesteps)
    
    # Add pump and dump schemes
    pump_points = np.random.choice(timesteps, size=5, replace=False)
    for point in pump_points:
        pump_size = np.random.uniform(1.5, 3.0)
        pump_duration = np.random.randint(10, 30)
        price[point:point+pump_duration] *= pump_size
        price[point+pump_duration:] *= 0.8  # Dump after pump
    
    # Simulate news events impact
    news_events = np.random.choice(timesteps, size=10, replace=False)
    for event in news_events:
        news_impact = np.random.uniform(-0.2, 0.2)
        price[event:] *= (1 + news_impact)
    
    # Add market sentiment analysis
    sentiment = np.random.normal(0, 0.1, timesteps)
    price += sentiment * price
    
    # Add sudden price jumps and drops
    jumps = np.zeros(timesteps)
    jump_points = np.random.choice(timesteps, size=20, replace=False)
    jumps[jump_points] = np.random.normal(0, 10, 20)
    
    # Combine all price components with noise
    price = 100 + trend + long_cycle + medium_cycle + short_cycle + \
           np.cumsum(jumps) + np.random.normal(0, 2, timesteps)
    
    # Volume: Create volume spikes and clusters
    base_volume = np.exp(np.random.normal(0, 0.5, timesteps))
    volume_spikes = np.zeros(timesteps)
    spike_points = np.random.choice(timesteps, size=50, replace=False)
    volume_spikes[spike_points] = np.random.exponential(2, 50)
    
    # Add volume clusters around price movements
    volume_clusters = np.zeros(timesteps)
    for i in range(1, timesteps-10):  # Ensure we have room for the full cluster
        if abs(price[i] - price[i-1]) > 1:
            cluster = np.random.exponential(1, 10)
            volume_clusters[i:i+10] += cluster
    
    volume = base_volume * (1 + volume_spikes + volume_clusters)
    
    # Liquidity: More complex pattern with periodic withdrawals
    base_liquidity = np.linspace(1000, 2000, timesteps)
    periodic_withdrawals = 300*np.sin(t/2) + 200*np.sin(t/4)
    liquidity_shocks = np.zeros(timesteps)
    shock_points = np.random.choice(timesteps, size=30, replace=False)
    liquidity_shocks[shock_points] = -np.random.exponential(200, 30)
        
    liquidity = base_liquidity + periodic_withdrawals + \
                np.cumsum(liquidity_shocks) + \
                300*np.random.normal(0, 1, timesteps)
    
    # Buys/Sells: More realistic trading patterns
    price_diff = np.diff(price, prepend=price[0])
    momentum = np.convolve(price_diff, np.ones(10)/10, mode='same')
    
    # Base buy/sell pressure
    buys = np.maximum(0, momentum + np.random.normal(0, 0.2, timesteps))
    sells = np.maximum(0, -momentum + np.random.normal(0, 0.2, timesteps))
    
    # Add trading bursts
    burst_points = np.random.choice(timesteps-5, size=40, replace=False)  # Ensure room for burst
    for point in burst_points:
        burst_size = min(5, timesteps - point)  # Prevent overflow
        if np.random.rand() > 0.5:
            buys[point:point+burst_size] += np.random.exponential(2, burst_size)
        else:
            sells[point:point+burst_size] += np.random.exponential(2, burst_size)
    
    # Combine and normalize
    dummy_data = np.column_stack([price, volume, liquidity, buys, sells])
        
    # Add additional features
    market_cap = np.cumsum(volume) * price  # Simple market cap calculation
    dummy_data = np.column_stack([dummy_data, market_cap])
        
    # Save normalization parameters
    normalization_params = {
        'mean': np.mean(dummy_data, axis=0),
        'std': np.std(dummy_data, axis=0)
    }
        
    # Normalize data
    dummy_data = (dummy_data - normalization_params['mean']) / normalization_params['std']
    
    # Save normalization parameters for later use
    np.save(os.path.join(token_manager.base_dir, 'normalization_params.npy'), normalization_params)

    # Save token data
    token_id = token_manager.get_token_id(token_address)
    sequences_path = os.path.join(token_manager.sequences_dir, f'token_{token_id}.h5')
    token_manager.save_token_sequences(token_address, dummy_data)
    print(f"Saved sequences for token {token_address} at {sequences_path}")
    
    # Verify file exists
    if os.path.exists(sequences_path):
        with h5py.File(sequences_path, 'r') as f:
            print(f"Verified H5 file. Shape: {f['sequences'].shape}")
    else:
        print(f"Warning: H5 file not found at {sequences_path}")
    
    # Create visualization directory for this token
    token_viz_dir = os.path.join(img_dir, f'token_{token_id}')
    os.makedirs(token_viz_dir, exist_ok=True)

    # Save visualizations for this token
    feature_names = ['Price', 'Volume', 'Liquidity', 'Buys', 'Sells']

    # Overall features plot
    fig, axes = plt.subplots(len(feature_names), 1, figsize=(12, 15))
    fig.suptitle(f'Token {token_id} Data Features', fontsize=16)

    for idx, (feature_name, ax) in enumerate(zip(feature_names, axes)):
        ax.plot(dummy_data[:, idx], label=feature_name)
        ax.set_title(feature_name)
        ax.set_xlabel('Time (15-second intervals)')
        ax.set_ylabel(feature_name)
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(token_viz_dir, 'features.png'))
    plt.close(fig)  # Close the figure explicitly
    
    # Save correlation heatmap
    corr_fig = plt.figure(figsize=(10, 8))
    correlation_matrix = np.corrcoef(dummy_data.T)
    sns.heatmap(correlation_matrix, annot=True, xticklabels=feature_names, yticklabels=feature_names)
    plt.title(f'Token {token_id} Feature Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(token_viz_dir, 'correlations.png'))
    plt.close(corr_fig)  # Close the figure explicitly with reference

    # Split into train/validation/test sets ensuring no data leakage
    train_idx = int(train_split * len(dummy_data))
    val_idx = train_idx + int(val_split * len(dummy_data))
    
    # Use different time periods for each split to avoid data leakage
    # We'll use sequential splits since this is time series data
    test_data = dummy_data[:train_idx]
    val_data = dummy_data[train_idx:val_idx]
    train_data = dummy_data[val_idx:]
    
    # Ensure no data leakage by removing overlapping sequences
    if args.seq_length > 0:
        # Remove the last sequence length from each split to avoid overlap
        if len(test_data) > args.seq_length:
            test_data = test_data[:-args.seq_length]
        if len(val_data) > args.seq_length:
            val_data = val_data[:-args.seq_length]
    
    print(f"Data split: Training={len(train_data)} examples, Validation={len(val_data)} examples, Testing={len(test_data)} examples")
    return train_data, val_data, test_data, dummy_data

# Set device for PyTorch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize token manager
token_manager = TokenDataManager(args.data_dir)

# Check if we just want to list available tokens
if args.list_tokens:
    print(f"Listing available tokens in {args.data_dir}:")
    
    # Check if metadata file exists
    if os.path.exists(token_manager.metadata_file):
        metadata = token_manager._load_metadata()
        print(f"Metadata file found: {token_manager.metadata_file}")
        
        if isinstance(metadata, dict):
            print(f"Available tokens from metadata:")
            for addr, token_id in metadata.items():
                print(f"  Token ID: {token_id}, Address: {addr}")
    
    # Check for token files
    if os.path.exists(token_manager.sequences_dir):
        files = os.listdir(token_manager.sequences_dir)
        token_files = [f for f in files if f.startswith('token_') and f.endswith('.h5')]
        print(f"\nAvailable token files in {token_manager.sequences_dir}:")
        for file in token_files:
            token_id = int(file.split('_')[1].split('.')[0])
            file_path = os.path.join(token_manager.sequences_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            
            # Get shape if possible
            try:
                with h5py.File(file_path, 'r') as f:
                    shape = f['sequences'].shape
                    print(f"  {file}: Token ID {token_id}, Shape {shape}, Size {file_size:.2f} MB")
            except:
                print(f"  {file}: Token ID {token_id}, Size {file_size:.2f} MB")
    
    # Exit after listing
    import sys
    sys.exit(0)

# Initialize variables
dummy_data = None
train_data = None
val_data = None
test_data = None
token_addresses = []
seq_len = args.seq_length

# Only generate token data if we're not in test mode
if not args.test_mode:
    if args.tokens_limit is not None:
        # Load existing tokens up to the specified limit
        metadata = token_manager._load_metadata()
        if isinstance(metadata, dict):
            token_addresses = list(metadata.keys())[:args.tokens_limit]
        elif isinstance(metadata, list) and len(metadata) > 0 and isinstance(metadata[0], dict):
            token_addresses = [list(item.keys())[0] for item in metadata[:args.tokens_limit]]
        else:
            raise ValueError("Metadata format not recognized. Expected a dictionary or list of dictionaries.")
            
        print(f"Using {len(token_addresses)} existing tokens for training")
        
        # Set timesteps from args when using existing tokens
        timesteps = args.timesteps
    else:
        def generate_token_addresses(num_tokens):
            """Generate a list of unique token addresses."""
            token_addresses = []
            for _ in range(num_tokens):
                token_address = '0x' + ''.join(np.random.choice(list('0123456789abcdef'), size=40))
                token_addresses.append(token_address)
            return token_addresses

        # Determine how many examples to generate
        total_examples = args.train_examples if args.train_examples is not None else args.timesteps * args.num_tokens
        print(f"Generating approximately {total_examples} total examples")

        # Calculate how many tokens and timesteps we need
        if args.train_examples is not None:
            # We'll generate one token with the exact number of timesteps needed
            num_tokens = 1
            timesteps = total_examples + args.seq_length  # Add sequence length to ensure we have enough data points
        else:
            num_tokens = args.num_tokens
            timesteps = args.timesteps

        token_addresses = generate_token_addresses(num_tokens)

    print(f"Generating data for {len(token_addresses)} tokens with {timesteps} timesteps each...")
    for token_address in token_addresses:
        train_data, val_data, test_data, current_dummy_data = generate_token_data(
            token_manager, 
            token_address, 
            timesteps=timesteps,
            train_split=0.4,
            val_split=0.2,
            test_split=0.4
        )
        # Keep the last token's data for prediction
        dummy_data = current_dummy_data

    # Ensure dummy_data is not None before proceeding
    if dummy_data is None:
        raise ValueError("No token data was generated. Please check the token generation process.")

    # Determine split strategy
    if args.split_strategy == 'token':
        # Split tokens into train/val/test sets
        np.random.shuffle(token_addresses)  # Shuffle to ensure random distribution
        
        num_train_tokens = max(1, int(args.train_token_ratio * len(token_addresses)))
        num_val_tokens = max(1, int((len(token_addresses) - num_train_tokens) / 2))
        
        train_tokens = token_addresses[:num_train_tokens]
        val_tokens = token_addresses[num_train_tokens:num_train_tokens+num_val_tokens]
        test_tokens = token_addresses[num_train_tokens+num_val_tokens:]
        
        print(f"Token-based split: {len(train_tokens)} training tokens, {len(val_tokens)} validation tokens, {len(test_tokens)} test tokens")
        
        # Create dataset using only training tokens
        forward_steps = 1
        train_dataset = MultiTokenDataset(
            token_manager,
            train_tokens,  # Only use training tokens
            seq_length=seq_len,
            forward_steps=forward_steps
        )
        
        # Save the token split information for reference
        split_info = {
            'strategy': 'token',
            'train_tokens': [token_manager.get_token_id(addr) for addr in train_tokens],
            'val_tokens': [token_manager.get_token_id(addr) for addr in val_tokens],
            'test_tokens': [token_manager.get_token_id(addr) for addr in test_tokens]
        }
        
    elif args.split_strategy == 'mixed':
        # Mixed strategy: Uses different tokens for different purposes
        # Some tokens are used entirely for training, others for validation, and others for testing
        # This helps evaluate how well the model generalizes to completely unseen tokens
        np.random.shuffle(token_addresses)  # Shuffle to ensure random distribution
        
        # Split tokens for training, validation and testing
        num_train_tokens = max(1, int(args.train_token_ratio * len(token_addresses)))
        num_val_tokens = max(1, int((len(token_addresses) - num_train_tokens) / 2))
        
        train_tokens = token_addresses[:num_train_tokens]
        val_tokens = token_addresses[num_train_tokens:num_train_tokens+num_val_tokens]
        test_tokens = token_addresses[num_train_tokens+num_val_tokens:]
        
        print(f"Mixed split: {len(train_tokens)} training tokens, {len(val_tokens)} validation tokens, {len(test_tokens)} test tokens")
        
        # Create dataset using only training tokens
        forward_steps = 1
        train_dataset = MultiTokenDataset(
            token_manager,
            train_tokens,  # Only use training tokens
            seq_length=seq_len,
            forward_steps=forward_steps
        )
        
        # For validation, use the validation tokens
        val_dataset = MultiTokenDataset(
            token_manager,
            val_tokens,  # Only use validation tokens
            seq_length=seq_len,
            forward_steps=forward_steps
        )
        
        # For testing, create datasets from the test tokens
        # We use the entire sequence of each test token to evaluate how well
        # the model generalizes to completely new tokens
        test_datasets = {}
        for token_address in test_tokens:
            token_id = token_manager.get_token_id(token_address)
            token_data = token_manager.load_token_sequences(token_address)
            
            # Create a dataset for this token's test data
            # We use the entire token data for testing (no internal splitting)
            if len(token_data) > seq_len:
                test_datasets[token_id] = TensorDataset(
                    torch.Tensor(token_data[:-seq_len]),  # Input sequences
                    torch.Tensor(token_data[seq_len:])    # Target sequences
                )
        
        # Save the token split information for reference
        split_info = {
            'strategy': 'mixed',
            'train_tokens': [token_manager.get_token_id(addr) for addr in train_tokens],
            'val_tokens': [token_manager.get_token_id(addr) for addr in val_tokens],
            'test_tokens': [token_manager.get_token_id(addr) for addr in test_tokens],
            'description': 'Different tokens used for training, validation, and testing to evaluate generalization to new tokens'
        }
        
    else:  # 'time' strategy
        # Use all tokens (current time-based approach)
        forward_steps = 1
        train_dataset = MultiTokenDataset(
            token_manager,
            token_addresses,
            seq_length=seq_len,
            forward_steps=forward_steps
        )
        
        # Save the token split information for reference
        split_info = {
            'strategy': 'time',
            'tokens': [token_manager.get_token_id(addr) for addr in token_addresses],
            'train_split': 0.4,
            'val_split': 0.2,
            'test_split': 0.4
        }
    
    # Save split info to a file
    import json
    with open(os.path.join(args.data_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Saved split information to {os.path.join(args.data_dir, 'split_info.json')}")

    # No need to limit training examples as we've already generated the exact amount needed
    print(f"Using all {len(train_dataset)} available training examples")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Print dataset information
    print(f"Dataset information:")
    print(f"  - Number of tokens: {len(token_addresses)}")
    print(f"  - Timesteps per token: {timesteps}")
    print(f"  - Total examples: {len(train_dataset)}")
    print(f"  - Sequence length: {args.seq_length}")
    print(f"  - Training batch size: {args.batch_size}")
    print(f"  - Device: {device}")
    print(f"  - Split strategy: {args.split_strategy}")
    if args.split_strategy == 'time':
        print(f"  - Data split: 40% training, 20% validation, 40% testing")
    else:
        print(f"  - Token split: {args.train_token_ratio*100:.0f}% tokens for training, remainder split between validation and testing")

# Function to save training data
def save_training_data(data, token_addresses, data_dir):
    """Save training data to disk"""
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, 'token_data.npy'), data)
    with open(os.path.join(data_dir, 'token_addresses.txt'), 'w') as f:
        for address in token_addresses:
            f.write(f"{address}\n")
    print(f"Training data saved to {data_dir}")

# Function to save model
def save_model(model, path):
    """Save model to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Get the model configuration
    # Use args.nhead since the model doesn't store it directly
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'seq_len': model.seq_len,
            'embed_size': model.embed_size,
            'nhead': args.nhead,  # Use the value from command-line args
            'dim_feedforward': model.transformer_encoder.linear1.out_features,
            'dropout': model.dropout.p,
            'device': model.device
        }
    }, path)
    print(f"Model saved to {path}")

# Function to load model
def load_model(path, device="cuda"):
    """Load model from disk"""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    
    # Update device based on current availability
    config['device'] = device
    
    model = ForecastingModel(
        seq_len=config['seq_len'],
        embed_size=config['embed_size'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        device=config['device']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

# Either load a pre-trained model or train a new one
if args.load_model:
    model = load_model(args.load_model, device)
    print(f"Loaded model from {args.load_model}")
else:
    import signal

    def save_model_on_interrupt(signum, frame):
        print("\nReceived interrupt signal. Saving model and exiting...")
        if args.save_model:
            save_model(model, args.save_model)
        import sys
        sys.exit(0)

    # Set up the signal handler
    signal.signal(signal.SIGINT, save_model_on_interrupt)

    import optuna
    from optuna.trial import Trial

    def objective(trial: Trial, args):
        # Define the hyperparameters to optimize
        embed_size = trial.suggest_categorical('embed_size', [64, 128, 256, 512])
        nhead = trial.suggest_int('nhead', 1, 8)
        seq_length = trial.suggest_int('seq_length', 100, 500, step=50)
        dim_feedforward = trial.suggest_int('dim_feedforward', 256, 2048, step=256)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8])
        epochs = args.epochs  # Use the epochs from command line args

        # Ensure embed_size is divisible by nhead
        while embed_size % nhead != 0:
            nhead = trial.suggest_int('nhead', 1, 8)

        # Training setup
        model = ForecastingModel(
            seq_len=seq_length, 
            embed_size=embed_size, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            device=device
        )
        model.to(device)
        model.train()
        criterion = torch.nn.HuberLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for xx, token_id, yy in train_loader:
                xx, yy = xx.to(device), yy.to(device)
                optimizer.zero_grad()
                out = model(xx)
                loss = criterion(out, yy)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            scheduler.step()
            avg_loss = epoch_loss / max(1, batch_count)
            trial.report(avg_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Return the final loss for optimization
        return avg_loss

    # Set up Optuna study
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    
    # Run the optimization
    study.optimize(lambda trial: objective(trial, args), n_trials=100)  # Adjust n_trials as needed

    # Print the best parameters
    print("Best hyperparameters:", study.best_params)
    print("Best trial value:", study.best_value)

    # Optionally, you can train the model with the best parameters found
    best_params = study.best_params
    model = ForecastingModel(
        seq_len=best_params['seq_length'],
        embed_size=best_params['embed_size'],
        nhead=best_params['nhead'],
        dim_feedforward=best_params['dim_feedforward'],
        dropout=best_params['dropout'],
        device=device
    )
    model.to(device)
    model.train()
    criterion = torch.nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'])
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("seq_length", best_params['seq_length'])
        mlflow.log_param("embed_size", best_params['embed_size'])
        mlflow.log_param("nhead", best_params['nhead'])
        mlflow.log_param("dim_feedforward", best_params['dim_feedforward'])
        mlflow.log_param("dropout", best_params['dropout'])
        mlflow.log_param("learning_rate", best_params['learning_rate'])
        mlflow.log_param("batch_size", best_params['batch_size'])
        mlflow.log_param("epochs", args.epochs)

        # Log metrics during training
        for epoch in range(args.epochs):
            # ... training code ...
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            if args.split_strategy == 'mixed':
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        # Log metrics during training
        for epoch in range(args.epochs):
            # ... training code ...
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            if args.split_strategy == 'mixed':
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        # Log metrics during training
        for epoch in range(args.epochs):
            # ... training code ...
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            if args.split_strategy == 'mixed':
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
    
    # Save training data if requested
    if args.save_training_data:
        # Save all token data
        for i, token_address in enumerate(token_addresses):
            token_id = token_manager.get_token_id(token_address)
            token_data = token_manager.load_token_sequences(token_address)
            save_training_data(token_data, [token_address], f"{args.data_dir}/token_{token_id}")
        print(f"Saved training data for {len(token_addresses)} tokens to {args.data_dir}")
    
    if args.split_strategy == 'mixed':
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Training loop with best parameters
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        batch_count = 0
        for xx, token_id, yy in train_loader:
            xx, yy = xx.to(device), yy.to(device)
            optimizer.zero_grad()
            out = model(xx)
            loss = criterion(out, yy)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        scheduler.step()
        avg_loss = epoch_loss / max(1, batch_count)
        print(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.6f}")
            
        # Log metrics
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        
        if args.split_strategy == 'mixed':
            # Validation step
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for xx, token_id, yy in val_loader:
                    xx, yy = xx.to(device), yy.to(device)
                    out = model(xx)
                    loss = criterion(out, yy)
                    val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                print(f"Epoch {epoch+1}/{args.epochs}: Validation Loss={avg_val_loss:.6f}")
    
    # Save the model if requested
    if args.save_model:
        save_model(model, args.save_model)
    
    # Log the model
    mlflow.pytorch.log_model(model, "model")
    
    # Log final metrics
    if args.split_strategy == 'mixed':
        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xx, token_id, yy in val_loader:
                xx, yy = xx.to(device), yy.to(device)
                out = model(xx)
                loss = criterion(out, yy)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss)

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

def run_prediction(model, data, seq_len, offset=None, forecast_steps=1000, extended_forecast_steps=3000, 
                   token_id=None, output_dir="./predictions", show_splits=False, split_strategy='time'):
    """
    Run prediction on data starting from a specific offset
    
    Args:
        model: The trained model
        data: The full dataset to use
        seq_len: Sequence length for the model
        offset: Starting point offset (if None, use the last seq_len points)
        forecast_steps: Number of steps to forecast for short-term prediction
        extended_forecast_steps: Number of additional steps for extended prediction
        token_id: Token ID for output file naming
        output_dir: Directory to save prediction outputs
        show_splits: Whether to show data splits in the visualization
        split_strategy: The data splitting strategy used ('time', 'token', or 'mixed')
    """
    """
    Run prediction on data starting from a specific offset
    
    Args:
        model: The trained model
        data: The full dataset to use
        seq_len: Sequence length for the model
        offset: Starting point offset (if None, use the last seq_len points)
        forecast_steps: Number of steps to forecast for short-term prediction
        extended_forecast_steps: Number of additional steps for extended prediction
        token_id: Token ID for output file naming
        output_dir: Directory to save prediction outputs
    """
    model.eval()
    
    # Determine the starting point for prediction
    if offset is None:
        # Use the last seq_len points by default
        start_idx = max(0, data.shape[0] - seq_len)
    else:
        # Use the specified offset
        start_idx = min(max(0, offset), data.shape[0] - seq_len)
    
    # Get the sequence to start prediction from
    prediction_start = start_idx + seq_len
    latest_data = data[start_idx:prediction_start]
    
    # For mixed strategy, we already have the test data from the token's dataset
    # No need to do additional splitting since we're testing on completely separate tokens
    if split_strategy == 'mixed':
        if token_id in test_datasets:
            # Just verify we have the data, but continue using the provided data parameter
            if test_datasets[token_id].tensors[0].shape[0] != data.shape[0]:
                print(f"Warning: Test dataset shape {test_datasets[token_id].tensors[0].shape} differs from provided data shape {data.shape}")
    
    # Make a copy to avoid modifying the original
    x = latest_data.copy()
    
    # Short-term prediction
    with torch.no_grad():
        for ff in range(forecast_steps):
            xx = x[-seq_len:]  # Use the last seq_len steps
            yy = model(torch.Tensor(xx).reshape(1, seq_len, 6).to(device))
            yy = yy.detach().cpu().numpy().reshape(1)
            
            # Create a new row with all features
            new_row = np.zeros(6)  # Create a row with zeros for all 6 features
            new_row[0] = yy[0]     # Set the first feature (price) to the predicted value
            
            # Add the new row to our sequence
            x = np.vstack([x, new_row])
    
    # Plot short-term predictions
    fig = plt.figure(figsize=(12, 7))
    
    # Determine how much historical data to show
    history_to_show = min(1000, start_idx)
    history_start = max(0, start_idx - history_to_show)
    
    # Plot historical data before prediction point
    plt.plot(range(history_start, prediction_start), 
             data[history_start:prediction_start, 0], 
             label="Historical", color='blue')
    
    # Plot actual data after prediction point (if available)
    if prediction_start < data.shape[0]:
        actual_end = min(data.shape[0], prediction_start + forecast_steps)
        plt.plot(range(prediction_start, actual_end),
                 data[prediction_start:actual_end, 0],
                 'g-', label="Actual", linewidth=2)
        
        # Highlight overlap between actual and predicted data
        overlap_end = min(actual_end, prediction_start + forecast_steps)
        plt.fill_between(range(prediction_start, overlap_end), 
                         data[prediction_start:overlap_end, 0], 
                         x[-(overlap_end - prediction_start):, 0], 
                         alpha=0.2, color='yellow', label='Overlap')
    
    # Plot predicted data
    plt.plot(range(prediction_start, prediction_start + forecast_steps), 
             x[-forecast_steps:, 0], 'r--', label="Predicted", linewidth=2)
    
    # Add a vertical line where prediction starts
    plt.axvline(x=prediction_start, color='black', linestyle='--', alpha=0.5)
    plt.text(prediction_start + 10, min(data[max(0, prediction_start-100):prediction_start, 0]), 
             'Prediction Start', rotation=90, verticalalignment='bottom')
    
    # Add data split lines if requested
    if show_splits:
        # Calculate split indices based on the same logic used in generate_token_data
        train_idx = int(0.4 * len(data))  # 40% for test
        val_idx = train_idx + int(0.2 * len(data))  # 20% for validation
        
        # Add vertical lines for data splits
        plt.axvline(x=train_idx, color='blue', linestyle=':', alpha=0.5)
        plt.text(train_idx + 10, min(data[:, 0]), 'Test/Val Split', 
                rotation=90, verticalalignment='bottom', color='blue')
                
        plt.axvline(x=val_idx, color='green', linestyle=':', alpha=0.5)
        plt.text(val_idx + 10, min(data[:, 0]), 'Val/Train Split', 
                rotation=90, verticalalignment='bottom', color='green')
    
    plt.title(f"Price Prediction (Short-term) - {'Token '+str(token_id) if token_id is not None else 'Model Test'}")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Price")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Save the figure
    token_prefix = f"token_{token_id}_" if token_id is not None else ""
    offset_str = f"offset_{offset}_" if offset is not None else ""
    short_term_filename = os.path.join(output_dir, f"{token_prefix}{offset_str}prediction_short.png")
    fig.savefig(short_term_filename)
    plt.close(fig)  # Close the figure to free memory
    print(f"Saved short-term prediction plot to {short_term_filename}")
    
    # Extended prediction
    with torch.no_grad():
        for ff in range(extended_forecast_steps):
            xx = x[-seq_len:]  # Use the last seq_len steps
            yy = model(torch.Tensor(xx).reshape(1, seq_len, 6).to(device))
            yy = yy.detach().cpu().numpy().reshape(1)
            
            # Create a new row with all features
            new_row = np.zeros(6)  # Create a row with zeros for all 6 features
            new_row[0] = yy[0]     # Set the first feature (price) to the predicted value
            
            # Add the new row to our sequence
            x = np.vstack([x, new_row])
    
    # Plot extended predictions
    fig = plt.figure(figsize=(14, 8))
    
    # Plot historical data before prediction point
    plt.plot(range(history_start, prediction_start), 
             data[history_start:prediction_start, 0], 
             label="Historical", color='blue')
    
    # Plot actual data after prediction point (if available)
    total_forecast = forecast_steps + extended_forecast_steps
    if prediction_start < data.shape[0]:
        actual_end = min(data.shape[0], prediction_start + total_forecast)
        plt.plot(range(prediction_start, actual_end),
                 data[prediction_start:actual_end, 0],
                 'g-', label="Actual", linewidth=2)
    
    # Plot predicted data
    plt.plot(range(prediction_start, prediction_start + total_forecast), 
             x[-total_forecast:, 0], 'r--', label="Predicted", linewidth=2)
    
    # Add a vertical line where prediction starts
    plt.axvline(x=prediction_start, color='black', linestyle='--', alpha=0.5)
    plt.text(prediction_start + 50, min(data[max(0, prediction_start-100):prediction_start, 0]), 
             'Prediction Start', rotation=90, verticalalignment='bottom')
    
    # Add data split lines if requested
    if show_splits:
        # Calculate split indices based on the same logic used in generate_token_data
        train_idx = int(0.4 * len(data))  # 40% for test
        val_idx = train_idx + int(0.2 * len(data))  # 20% for validation
        
        # Add vertical lines for data splits
        plt.axvline(x=train_idx, color='blue', linestyle=':', alpha=0.5)
        plt.text(train_idx + 50, min(data[:, 0]), 'Test/Val Split', 
                rotation=90, verticalalignment='bottom', color='blue')
                
        plt.axvline(x=val_idx, color='green', linestyle=':', alpha=0.5)
        plt.text(val_idx + 50, min(data[:, 0]), 'Val/Train Split', 
                rotation=90, verticalalignment='bottom', color='green')
    
    # Add a vertical line where short-term forecast ends
    plt.axvline(x=prediction_start + forecast_steps, color='gray', linestyle=':', alpha=0.5)
    plt.text(prediction_start + forecast_steps + 50, 
             min(data[max(0, prediction_start-100):prediction_start, 0]), 
             'Extended Forecast Start', rotation=90, verticalalignment='bottom')
    
    plt.title(f"Price Prediction (Extended) - {'Token '+str(token_id) if token_id is not None else 'Model Test'}")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    extended_filename = os.path.join(output_dir, f"{token_prefix}{offset_str}prediction_extended.png")
    fig.savefig(extended_filename)
    plt.close(fig)  # Close the figure to free memory
    print(f"Saved extended prediction plot to {extended_filename}")
    
    return x

# Function to visualize data splits
def visualize_data_splits(data, token_id, output_dir, offset=None):
    """Visualize the data splits for training, validation, and testing with optional offset"""
    if offset is not None:
        data = data[offset:]
    
    # Calculate split indices based on the same logic used in generate_token_data
    train_idx = int(0.4 * len(data))  # 40% for test
    val_idx = train_idx + int(0.2 * len(data))  # 20% for validation
    
    plt.figure(figsize=(12, 6))
    
    # Plot the entire dataset
    if offset is not None:
        plt.plot(range(offset, offset + len(data)), data[:, 0], 'gray', alpha=0.3, label='All Data')
    else:
        plt.plot(range(len(data)), data[:, 0], 'gray', alpha=0.3, label='All Data')
    
    # Plot the test data (first portion)
    if offset is not None:
        plt.plot(range(offset, offset + train_idx), data[:train_idx, 0], 'b', label='Test Data')
    else:
        plt.plot(range(0, train_idx), data[:train_idx, 0], 'b', label='Test Data')
    
    # Plot the validation data (middle portion)
    if offset is not None:
        plt.plot(range(offset + train_idx, offset + val_idx), data[train_idx:val_idx, 0], 'g', label='Validation Data')
    else:
        plt.plot(range(train_idx, val_idx), data[train_idx:val_idx, 0], 'g', label='Validation Data')
    
    # Plot the training data (last portion)
    if offset is not None:
        plt.plot(range(offset + val_idx, offset + len(data)), data[val_idx:, 0], 'r', label='Training Data')
    else:
        plt.plot(range(val_idx, len(data)), data[val_idx:, 0], 'r', label='Training Data')
    
    plt.title(f'Data Splits for Token {token_id}')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    split_filename = os.path.join(output_dir, f'token_{token_id}_data_splits.png')
    plt.savefig(split_filename)
    plt.close()  # Close the figure to free memory
    print(f"Data splits visualization saved to {split_filename}")

# Function to get token data by ID
def get_token_data_by_id(token_id, token_manager, metadata=None):
    """Get token data for a specific token ID"""
    if metadata is None:
        if not os.path.exists(token_manager.metadata_file):
            raise ValueError(f"No token metadata found at {token_manager.metadata_file}. Make sure you're using the correct data directory.")
        metadata = token_manager._load_metadata()
    
    # Find the token address for this ID
    token_addresses = []
    found = False
    
    if isinstance(metadata, dict):
        # Check if metadata has a nested structure with 'token_ids'
        if 'token_ids' in metadata:
            for addr, tid in metadata['token_ids'].items():
                if tid == token_id:
                    token_addresses.append(addr)
                    found = True
                    break
        else:
            # Standard dictionary format
            for addr, tid in metadata.items():
                if tid == token_id:
                    token_addresses.append(addr)
                    found = True
                    break
    elif isinstance(metadata, list) and len(metadata) > 0 and isinstance(metadata[0], dict):
        # List containing dictionary format
        for item in metadata:
            if isinstance(item, dict):
                for addr, tid in item.items():
                    if tid == token_id:
                        token_addresses.append(addr)
                        found = True
                        break
                if found:
                    break
    
    if not found:
        # Try to load directly from file without using metadata
        token_file = os.path.join(token_manager.sequences_dir, f'token_{token_id}.h5')
        if os.path.exists(token_file):
            # Load directly from file
            with h5py.File(token_file, 'r') as f:
                data = f['sequences'][:]
            return data, None
        
        # If we get here, we couldn't find the token
        raise ValueError(f"No token found with ID {token_id}")
    
    # Check if the token file exists
    token_file = os.path.join(token_manager.sequences_dir, f'token_{token_id}.h5')
    if not os.path.exists(token_file):
        raise ValueError(f"Token data file not found at {token_file}")
    
    # Load the token data
    token_address = token_addresses[0]
    data = token_manager.load_token_sequences(token_address)
    return data, token_address

# Check if we're in test mode
if args.test_mode:
    if args.load_model is None:
        raise ValueError("Test mode requires a model to be loaded with --load-model")
    
    # Load the model
    model = load_model(args.load_model, device)
    print(f"Loaded model from {args.load_model} for testing")
    
    # Get available token IDs from files for validation
    available_token_ids = []
    if os.path.exists(token_manager.sequences_dir):
        files = os.listdir(token_manager.sequences_dir)
        token_files = [f for f in files if f.startswith('token_') and f.endswith('.h5')]
        available_token_ids = [int(f.split('_')[1].split('.')[0]) for f in token_files]
    
    # Load metadata once to avoid repeated loading
    metadata = None
    if os.path.exists(token_manager.metadata_file):
        metadata = token_manager._load_metadata()
    
    # Process token range if specified
    token_ids_to_test = []
    
    if args.test_token_range is not None:
        try:
            start, end = map(int, args.test_token_range.split('-'))
            token_ids_to_test = list(range(start, end + 1))
            print(f"Testing on token range: {start} to {end}")
            
            # Validate token IDs
            for token_id in token_ids_to_test:
                if token_id not in available_token_ids:
                    print(f"Warning: Token ID {token_id} not found in available files. It will be skipped.")
            
            # Filter to only include available tokens
            token_ids_to_test = [tid for tid in token_ids_to_test if tid in available_token_ids]
            
            if not token_ids_to_test:
                raise ValueError(f"No tokens found in the specified range {args.test_token_range}. Available token IDs: {sorted(available_token_ids)}")
        
        except ValueError as e:
            if "not enough values to unpack" in str(e):
                raise ValueError(f"Invalid token range format. Use start-end format (e.g., 0-5)")
            else:
                raise e
    
    # Single token ID specified
    elif args.test_token_id is not None:
        token_ids_to_test = [args.test_token_id]
    
    # No token specified, use the last generated token
    else:
        if dummy_data is None:
            raise ValueError("No token data available. Please specify a token ID with --test-token-id or --test-token-range")
        test_data = dummy_data
        token_id = token_manager.get_token_id(token_addresses[-1])
        token_ids_to_test = [token_id]
        print(f"Testing on last generated token (ID: {token_id})")
    
    # Process each token
    for token_id in token_ids_to_test:
        try:
            print(f"\n--- Testing on token ID {token_id} ---")
            
            # Get token data
            test_data, token_address = get_token_data_by_id(token_id, token_manager, metadata)
            
            if token_address:
                print(f"Testing on token ID {token_id} with address {token_address}")
            else:
                print(f"Testing on token ID {token_id} (loaded directly from file)")
            
            # Generate data splits visualization if requested
            if args.show_data_splits:
                visualize_data_splits(test_data, token_id, args.output_dir)
            
            # Run prediction with the specified offset
            run_prediction(
                model=model,
                data=test_data,
                seq_len=seq_len,
                offset=args.test_offset,
                forecast_steps=args.forecast,
                extended_forecast_steps=args.forecast_extended,
                token_id=token_id,
                output_dir=args.output_dir,
                show_splits=args.show_data_splits
            )
            
        except Exception as e:
            print(f"Error processing token ID {token_id}: {str(e)}")
            print("Continuing with next token...")
    
    # Exit if we're only testing
    import sys
    sys.exit(0)

# If not in test mode, continue with normal training and prediction
# Prediction Loop
FORCAST = args.forecast
model.eval()
    
if args.split_strategy == 'mixed':
    # Test each token's test data
    for token_id, test_dataset in test_datasets.items():
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_loss = 0.0
        with torch.no_grad():
            for xx, yy in test_loader:
                xx, yy = xx.to(device), yy.to(device)
                out = model(xx)
                loss = criterion(out, yy)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(f"Token {token_id} Test Loss={avg_test_loss:.6f}")
            
        # Run prediction on this token's test data
        run_prediction(
            model=model,
            data=test_dataset.tensors[0].numpy(),
            seq_len=seq_len,
            offset=None,  # Use the last seq_len points
            forecast_steps=args.forecast,
            extended_forecast_steps=args.forecast_extended,
            token_id=token_id,
            output_dir=args.output_dir,
            show_splits=args.show_data_splits,
            split_strategy=args.split_strategy
        )
else:
    # Use the last token's data for prediction
    if dummy_data is None:
        raise ValueError("dummy_data is None. Cannot make predictions.")

    # Run prediction on the training data
    run_prediction(
        model=model,
        data=dummy_data,
        seq_len=seq_len,
        offset=None,  # Use the last seq_len points
        forecast_steps=args.forecast,
        extended_forecast_steps=args.forecast_extended,
        token_id=None,
        output_dir=args.output_dir,  # Use the specified output directory
        show_splits=args.show_data_splits,
        split_strategy=args.split_strategy
    )
    if args.show_data_splits:
        visualize_data_splits(dummy_data, None, args.output_dir, offset=None)
