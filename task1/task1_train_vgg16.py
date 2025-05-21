import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import json
import matplotlib.pyplot as plt

# --- Import modules ---
from data.loaders import get_cifar_loader
# Import the new model (assuming it's saved in models/res_vgg.py or similar)
from models.vgg16 import ResVGG16 # Adjust the import path if needed
# Or if added to models/vgg.py: from models.vgg import ResVGG16

# =============================================================================
# 0. Configuration
# =============================================================================
# --- Basic Config ---
BATCH_SIZE = 128
EPOCHS = 60           # Increased further for potentially deeper model
DATA_DIR = './data/'
NUM_WORKERS = 4
SEED = 42

# --- Paths ---
# Using the same paths as before, but results will be for the new model
RESULTS_DIR = "/remote-home/yinzhitao/pj2/codes/VGG_BatchNorm/task1_train_res"
FIGURES_DIR = "/remote-home/yinzhitao/pj2/codes/VGG_BatchNorm/task1_train_fig"
MODEL_SAVE_DIR = os.path.join(RESULTS_DIR, "models_ResVGG16") # Specific subdir for this model type

# --- Experiment Configurations (Focus on Hyperparameters) ---
# Base setup: CrossEntropyLoss + SGD optimizer
BASE_LOSS_FN = nn.CrossEntropyLoss
BASE_LOSS_NAME = "CrossEntropy"
BASE_OPTIMIZER = optim.SGD
BASE_OPT_NAME = "SGD"

# Hyperparameters to test for SGD: Learning Rate, Weight Decay
HYPERPARAM_CONFIGS = [
    {"opt_params": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4}},
    {"opt_params": {"lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4}},
    {"opt_params": {"lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4}},
    {"opt_params": {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-3}}, # Different weight decay
    {"opt_params": {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4}}, # Different weight decay
    # {"opt_params": {"lr": 0.1, "momentum": 0.95, "weight_decay": 5e-4}}, # Different momentum (less common to tune)
    # Add more hyperparameter combinations for SGD
    # Or, choose Adam as base and vary its lr, weight_decay (AdamW), betas
    # Example with AdamW:
    # {"optimizer": optim.AdamW, "opt_name": "AdamW", "opt_params": {"lr": 0.001, "weight_decay": 0.01}},
    # {"optimizer": optim.AdamW, "opt_name": "AdamW", "opt_params": {"lr": 0.0005, "weight_decay": 0.01}},
]

# Learning Rate Scheduler (Optional but Recommended for SGD)
# Example: StepLR or CosineAnnealingLR
USE_SCHEDULER = True

# Set random seed
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# =============================================================================
# 1. Plotting Function (Keep as before)
# =============================================================================
def plot_and_save_curves(train_losses, val_losses, val_accuracies, run_name, save_dir):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{run_name} - Loss Curves')
    plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)')
    plt.title(f'{run_name} - Validation Accuracy')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{run_name}_curves.png")
    plt.savefig(save_path); plt.close()

# =============================================================================
# 2. Training Function (Add Scheduler Step)
# =============================================================================
def train_model(model, trainloader, valloader, criterion, optimizer, scheduler, device, epochs, model_save_path, run_name):
    model.to(device)
    best_run_val_accuracy = 0.0
    train_losses, val_losses, val_accuracies = [], [], []
    print(f"[{run_name}] Starting training for {epochs} epochs...")
    run_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        samples_processed = 0
        for i, data in enumerate(trainloader): # Can add tqdm here if desired
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            samples_processed += inputs.size(0)

        epoch_train_loss = running_loss / samples_processed
        train_losses.append(epoch_train_loss)

        epoch_val_loss, epoch_val_accuracy = evaluate_model(model, valloader, criterion, device)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        epoch_duration = time.time() - epoch_start_time
        print(f"[{run_name}] Epoch [{epoch + 1}/{epochs}] - {epoch_duration:.2f}s - "
              f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%")

        # Step the scheduler AFTER validation (common practice)
        if scheduler:
            scheduler.step()

        if epoch_val_accuracy > best_run_val_accuracy:
            best_run_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), model_save_path)

    run_duration = time.time() - run_start_time
    print(f"[{run_name}] Finished Training in {run_duration:.2f} seconds. Best Val Acc: {best_run_val_accuracy:.2f}%")
    return train_losses, val_losses, val_accuracies, best_run_val_accuracy

# =============================================================================
# 3. Evaluation Function (Keep as before)
# =============================================================================
def evaluate_model(model, dataloader, criterion, device):
    # ... (same as before) ...
    model.eval()
    total_loss = 0.0; correct_predictions = 0; total_samples = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs); loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0); correct_predictions += (predicted == labels).sum().item()
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy

# =============================================================================
# 4. Main Execution Block
# =============================================================================
if __name__ == '__main__':
    if not torch.cuda.is_available(): print("Error: CUDA not available."); exit()
    device = torch.device("cuda"); print(f"Using device: {device}")
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Create specific model dir
        print(f"Results base dir: {RESULTS_DIR}\nFigures base dir: {FIGURES_DIR}\nModels base dir: {MODEL_SAVE_DIR}")
    except OSError as e: print(f"Error creating directories: {e}"); exit()

    print("\nLoading CIFAR-10 data..."); # Add error handling if needed
    train_loader = get_cifar_loader(root=DATA_DIR, batch_size=BATCH_SIZE, train=True, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = get_cifar_loader(root=DATA_DIR, batch_size=BATCH_SIZE, train=False, shuffle=False, num_workers=NUM_WORKERS)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print("Data loaded.")

    all_run_summaries = []
    print(f"\nStarting {len(HYPERPARAM_CONFIGS)} experiments with ResVGG16...")

    for i, config in enumerate(HYPERPARAM_CONFIGS):
        run_log_prefix = f"[Run {i+1}/{len(HYPERPARAM_CONFIGS)}]"
        opt_params_str = "_".join([f"{k}{v}" for k, v in config['opt_params'].items()])
        run_name = f"ResVGG16_{BASE_LOSS_NAME}_{BASE_OPT_NAME}_{opt_params_str}" # Use base names + changing params
        model_save_path = os.path.join(MODEL_SAVE_DIR, f"{run_name}_best_ep{EPOCHS}.pth")
        figure_save_path = os.path.join(FIGURES_DIR, f"{run_name}_curves_ep{EPOCHS}.png")

        print(f"\n{run_log_prefix} --- Configuration ---")
        print(f"  Model: ResVGG16")
        print(f"  Loss Function: {BASE_LOSS_NAME}")
        print(f"  Optimizer: {BASE_OPT_NAME}")
        print(f"  Optimizer Params: {config['opt_params']}")
        print(f"  Scheduler Used: {USE_SCHEDULER}")

        # 1. Initialize Model
        model = ResVGG16(num_classes=len(classes), dropout_p=0.5).to(device) # dropout_p can also be varied
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model Params: {num_params:,}")

        # 2. Initialize Loss, Optimizer, Scheduler
        criterion = BASE_LOSS_FN()
        optimizer = BASE_OPTIMIZER(model.parameters(), **config['opt_params'])
        scheduler = None
        if USE_SCHEDULER:
             # Example: Cosine Annealing. Adjust T_max based on EPOCHS.
             scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
             # Or StepLR: scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
             print(f"  Scheduler: {type(scheduler).__name__}")


        # 3. Train
        history = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, EPOCHS, model_save_path, run_name)
        train_losses, val_losses, val_accuracies, run_best_val_acc = history

        # 4. Process Results
        best_epoch_num = val_accuracies.index(run_best_val_acc) + 1 if run_best_val_acc > 0 else -1
        run_summary = {
            "run_name": run_name, "config_index": i+1, "model_type": "ResVGG16",
            "loss_function": BASE_LOSS_NAME, "optimizer": BASE_OPT_NAME,
            "optimizer_params": config['opt_params'], "scheduler_used": USE_SCHEDULER,
            "epochs_trained": EPOCHS, "best_validation_accuracy": run_best_val_acc,
            "best_epoch": best_epoch_num,
            "saved_model_path": model_save_path if run_best_val_acc > 0 else None,
            "figure_path": figure_save_path
        }
        all_run_summaries.append(run_summary)
        print(f"{run_log_prefix} --- Summary ---")
        print(f"  Best Val Acc: {run_best_val_acc:.2f}% at Epoch {best_epoch_num}")

        # 5. Plot Curves
        plot_and_save_curves(train_losses, val_losses, val_accuracies, run_name, FIGURES_DIR)
        print(f"  Training curves saved: {figure_save_path}")


    # --- Save Overall Summary ---
    summary_file_path = os.path.join(RESULTS_DIR, f"ResVGG16_experiments_summary_ep{EPOCHS}.json")
    print(f"\n--- All Experiments Finished ---")
    try:
        with open(summary_file_path, 'w') as f: json.dump(all_run_summaries, f, indent=4)
        print(f"Overall summary saved to: {summary_file_path}")
    except Exception as e: print(f"Error saving overall summary: {e}")

    # Print Top Results
    if all_run_summaries:
        all_run_summaries.sort(key=lambda x: x['best_validation_accuracy'], reverse=True)
        print("\n--- Top Performing Hyperparameters ---")
        for rank, summary in enumerate(all_run_summaries[:5]):
             opt_params_str = "_".join([f"{k}{v}" for k, v in summary['optimizer_params'].items()])
             print(f"  Rank {rank+1}: Opt={summary['optimizer']}, Params={opt_params_str} -> Best Val Acc: {summary['best_validation_accuracy']:.2f}% (Epoch {summary['best_epoch']})")