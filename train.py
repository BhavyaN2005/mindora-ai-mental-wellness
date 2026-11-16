"""
train.py
Training pipeline for the NumPy LSTM Intent Classifier.
STRICT CONSTRAINT: ONLY NumPy, Matplotlib, Pandas, and Python standard libraries.
"""

import os
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt

from preprocess import Preprocessor
from model import NumPyLSTM, AdamOptimizer


def train_model(
    data_path: str = "data/intents.json",
    vocab_path: str = "data/vocabulary.json",
    tags_path: str = "data/tags.json",
    model_save_path: str = "models/lstm_weights.npz",
    plot_save_path: str = "outputs/training_loss.png",
    embedding_dim: int = 64,
    hidden_dim: int = 64,
    epochs: int = 160,
    batch_size: int = 32,
    learning_rate: float = 0.007,
    seed: int = 42
):
    print("=" * 60)
    print("Starting NumPy LSTM Intent Classification Training")
    print("=" * 60)

    # 1. Preprocessing and metadata creation
    print(f"[1/6] Loading and preprocessing dataset from: {data_path}")
    preprocessor = Preprocessor()
    num_samples, vocab_size, num_classes = preprocessor.fit_from_json(data_path)
    
    print(f"      Total valid samples : {num_samples}")
    print(f"      Vocabulary size     : {vocab_size} (including <PAD>, <UNK>)")
    print(f"      Number of intents   : {num_classes}")

    preprocessor.save_metadata(vocab_path, tags_path)
    print(f"      Saved vocabulary to : {vocab_path}")
    print(f"      Saved tags to       : {tags_path}")

    # 2. Prepare numerical datasets
    print("[2/6] Preparing sequences and target arrays...")
    X_list, y_list, lengths_list = preprocessor.prepare_dataset(data_path)
    X = np.array(X_list, dtype=np.int32)
    y = np.array(y_list, dtype=np.int32)
    lengths = np.array(lengths_list, dtype=np.int32)
    N = len(X)

    # 3. Model & Optimizer initialization
    print(f"[3/6] Initializing NumPy LSTM model...")
    print(f"      Embedding Dim: {embedding_dim}, Hidden Dim: {hidden_dim}, Classes: {num_classes}")
    model = NumPyLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        seed=seed
    )
    optimizer = AdamOptimizer(
        params=model.params,
        lr=learning_rate,
        clip_norm=5.0
    )

    # 4. Training Loop
    print(f"[4/6] Training for {epochs} epochs (Batch size: {batch_size}, LR: {learning_rate})...")
    history_loss = []
    history_acc = []
    rng = np.random.RandomState(seed)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Shuffle dataset each epoch
        indices = rng.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        lengths_shuffled = lengths[indices]

        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = int(np.ceil(N / batch_size))

        for b_idx in range(num_batches):
            start_i = b_idx * batch_size
            end_i = min(start_i + batch_size, N)

            X_b = X_shuffled[start_i:end_i]
            y_b = y_shuffled[start_i:end_i]
            len_b = lengths_shuffled[start_i:end_i]

            # Forward pass
            probs_b, cache_b = model.forward(X_b, len_b)
            loss_b, acc_b = model.compute_loss(probs_b, y_b)

            # Backward pass (BPTT)
            grads_b = model.backward(cache_b, y_b)

            # Optimization step
            optimizer.step(model.params, grads_b)

            epoch_loss += loss_b * (end_i - start_i)
            epoch_acc += acc_b * (end_i - start_i)

        avg_loss = epoch_loss / N
        avg_acc = epoch_acc / N

        history_loss.append(avg_loss)
        history_acc.append(avg_acc)

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"      Epoch {epoch:3d}/{epochs:3d} - Loss: {avg_loss:.4f} - Accuracy: {avg_acc * 100:.2f}%")

    elapsed = time.time() - start_time
    print(f"      Training completed in {elapsed:.2f} seconds.")

    # 5. Save model weights
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_weights(model_save_path)
    print(f"[5/6] Saved trained LSTM weights to: {model_save_path}")

    # 6. Plot Loss and Accuracy curve
    print(f"[6/6] Generating training visualization plot...")
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history_loss, color="#2563EB", linewidth=2, label="Categorical Cross-Entropy")
    plt.title("NumPy LSTM Training Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history_acc], color="#059669", linewidth=2, label="Accuracy (%)")
    plt.title("NumPy LSTM Training Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=200)
    plt.close()
    print(f"      Saved visualization to: {plot_save_path}")

    print("=" * 60)
    print(f"Final Model Performance: Loss = {history_loss[-1]:.4f}, Accuracy = {history_acc[-1] * 100:.2f}%")
    print("=" * 60)

    return model, preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NumPy LSTM Mental Health Intent Classifier")
    parser.add_argument("--epochs", type=int, default=160, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.007, help="Learning rate for Adam")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
