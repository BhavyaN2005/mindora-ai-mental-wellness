"""
model.py
NumPy-based LSTM from scratch for Intent Classification.
STRICT CONSTRAINT: ONLY NumPy and standard Python libraries.
NO PyTorch, TensorFlow, Keras, Scikit-learn, etc.

Architecture:
- Word Embedding Lookup Table (learnable)
- LSTM Cell with 4 Gates: Forget, Input, Candidate, Output
- Backpropagation Through Time (BPTT)
- Dense Linear Classifier + Softmax
- Categorical Cross-Entropy Loss
- Adam Optimizer with Gradient Clipping
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    x_clipped = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class AdamOptimizer:
    """
    Adam optimizer implemented in pure NumPy with global gradient clipping.
    """
    def __init__(self, params: Dict[str, np.ndarray], lr: float = 0.005,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, clip_norm: float = 5.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_norm = clip_norm
        self.t = 0
        
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        self.t += 1
        
        # 1. Global gradient clipping
        total_sq_norm = 0.0
        for k, g in grads.items():
            total_sq_norm += np.sum(np.square(g))
        total_norm = np.sqrt(total_sq_norm)

        clip_factor = 1.0
        if total_norm > self.clip_norm:
            clip_factor = self.clip_norm / (total_norm + 1e-7)

        # 2. Adam parameter update
        for k in params:
            if k not in grads:
                continue
            g = grads[k] * clip_factor

            # Update biased 1st and 2nd moment estimates
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * np.square(g)

            # Compute bias-corrected estimates
            m_hat = self.m[k] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1.0 - self.beta2 ** self.t)

            # Update parameter
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class NumPyLSTM:
    """
    Full word-level LSTM Neural Network built entirely with NumPy.
    Includes forward pass, BPTT, categorical cross entropy, and persistence.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 64,
                 hidden_dim: int = 64, output_dim: int = 80,
                 seed: int = 42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seed = seed
        
        self.params: Dict[str, np.ndarray] = {}
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier / Glorot initialization."""
        rng = np.random.RandomState(self.seed)
        H = self.hidden_dim
        D = self.embedding_dim
        V = self.vocab_size
        C = self.output_dim

        # 1. Embedding Matrix (V, D)
        # Scale by standard normal scaled
        self.params["W_emb"] = rng.randn(V, D).astype(np.float32) * 0.1
        # Set PAD token embedding to 0
        self.params["W_emb"][0] = 0.0

        # 2. LSTM Weights: W_xh (D, 4H), W_hh (H, 4H), b_h (4H,)
        # Order of 4 gates: [Forget, Input, Candidate, Output]
        limit_xh = np.sqrt(6.0 / (D + 4 * H))
        self.params["W_xh"] = rng.uniform(-limit_xh, limit_xh, (D, 4 * H)).astype(np.float32)

        limit_hh = np.sqrt(6.0 / (H + 4 * H))
        self.params["W_hh"] = rng.uniform(-limit_hh, limit_hh, (H, 4 * H)).astype(np.float32)

        # Initialize biases; set Forget Gate bias to +1.0 for better gradient flow
        self.params["b_h"] = np.zeros((4 * H,), dtype=np.float32)
        self.params["b_h"][0:H] = 1.0  # Forget gate bias

        # 3. Dense Classifier: W_hy (H, C), b_y (C,)
        limit_hy = np.sqrt(6.0 / (H + C))
        self.params["W_hy"] = rng.uniform(-limit_hy, limit_hy, (H, C)).astype(np.float32)
        self.params["b_y"] = np.zeros((C,), dtype=np.float32)

    def forward(self, X: np.ndarray, lengths: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward propagation through Embedding, LSTM timesteps, and Dense Softmax.
        
        Args:
            X: Array of token indices of shape (B, T)
            lengths: Array of valid lengths of shape (B,)
            
        Returns:
            probs: Predicted probabilities of shape (B, C)
            cache: Dictionary of forward activations needed for backprop
        """
        B, T = X.shape
        H = self.hidden_dim
        D = self.embedding_dim

        # Ensure lengths are within [1, T]
        lengths = np.clip(lengths, 1, T)

        # Storage for timesteps
        xs = np.zeros((B, T, D), dtype=np.float32)
        hs = np.zeros((B, T, H), dtype=np.float32)
        cs = np.zeros((B, T, H), dtype=np.float32)
        
        fs = np.zeros((B, T, H), dtype=np.float32)
        iss = np.zeros((B, T, H), dtype=np.float32)
        gs = np.zeros((B, T, H), dtype=np.float32)
        os = np.zeros((B, T, H), dtype=np.float32)
        c_t_tanhs = np.zeros((B, T, H), dtype=np.float32)

        h_prev = np.zeros((B, H), dtype=np.float32)
        c_prev = np.zeros((B, H), dtype=np.float32)

        W_xh = self.params["W_xh"]
        W_hh = self.params["W_hh"]
        b_h = self.params["b_h"]
        W_emb = self.params["W_emb"]

        for t in range(T):
            token_ids = X[:, t]
            x_t = W_emb[token_ids]  # (B, D)
            xs[:, t, :] = x_t

            # Linear transformation for all 4 gates
            z = np.dot(x_t, W_xh) + np.dot(h_prev, W_hh) + b_h  # (B, 4H)

            # Gate activations
            f_t = sigmoid(z[:, 0:H])
            i_t = sigmoid(z[:, H:2*H])
            g_t = np.tanh(z[:, 2*H:3*H])
            o_t = sigmoid(z[:, 3*H:4*H])

            # Update cell state and hidden state
            c_t = f_t * c_prev + i_t * g_t
            c_tanh = np.tanh(c_t)
            h_t = o_t * c_tanh

            # Cache values for timestep t
            fs[:, t, :] = f_t
            iss[:, t, :] = i_t
            gs[:, t, :] = g_t
            os[:, t, :] = o_t
            cs[:, t, :] = c_t
            c_t_tanhs[:, t, :] = c_tanh
            hs[:, t, :] = h_t

            h_prev = h_t
            c_prev = c_t

        # Extract sequence representation using the hidden state at the last valid token
        h_rep = np.zeros((B, H), dtype=np.float32)
        for b in range(B):
            valid_idx = lengths[b] - 1
            h_rep[b] = hs[b, valid_idx, :]

        # Dense classification layer
        W_hy = self.params["W_hy"]
        b_y = self.params["b_y"]
        logits = np.dot(h_rep, W_hy) + b_y  # (B, C)

        # Numerically stable Softmax
        shift_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shift_logits)
        probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-12)

        cache = {
            "X": X,
            "lengths": lengths,
            "xs": xs,
            "hs": hs,
            "cs": cs,
            "fs": fs,
            "iss": iss,
            "gs": gs,
            "os": os,
            "c_t_tanhs": c_t_tanhs,
            "h_rep": h_rep,
            "probs": probs
        }
        return probs, cache

    def compute_loss(self, probs: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Computes Categorical Cross-Entropy loss and accuracy.
        
        Args:
            probs: (B, C)
            y: (B,) class labels
        Returns:
            loss: float
            accuracy: float (0.0 to 1.0)
        """
        B = probs.shape[0]
        correct_probs = probs[np.arange(B), y]
        log_probs = np.log(np.clip(correct_probs, 1e-15, 1.0))
        loss = -float(np.mean(log_probs))

        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y))
        return loss, acc

    def backward(self, cache: Dict[str, Any], y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backpropagation Through Time (BPTT).
        Computes analytical gradients for all network weights.
        """
        X = cache["X"]
        lengths = cache["lengths"]
        xs = cache["xs"]
        hs = cache["hs"]
        cs = cache["cs"]
        fs = cache["fs"]
        iss = cache["iss"]
        gs = cache["gs"]
        os = cache["os"]
        c_t_tanhs = cache["c_t_tanhs"]
        h_rep = cache["h_rep"]
        probs = cache["probs"]

        B, T = X.shape
        H = self.hidden_dim
        D = self.embedding_dim
        V = self.vocab_size

        W_xh = self.params["W_xh"]
        W_hh = self.params["W_hh"]
        W_hy = self.params["W_hy"]

        # 1. Gradient wrt Softmax Logits: dL/d(logits) = (probs - y_onehot) / B
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= float(B)

        # 2. Gradients for output classifier
        dW_hy = np.dot(h_rep.T, dlogits)
        db_y = np.sum(dlogits, axis=0)
        dh_rep = np.dot(dlogits, W_hy.T)  # (B, H)

        # 3. Distribute gradient to the chosen final hidden state of each sequence
        dhs = np.zeros((B, T, H), dtype=np.float32)
        for b in range(B):
            valid_idx = lengths[b] - 1
            dhs[b, valid_idx, :] += dh_rep[b]

        # 4. Backpropagation through LSTM timesteps
        dW_xh = np.zeros_like(W_xh)
        dW_hh = np.zeros_like(W_hh)
        db_h = np.zeros_like(self.params["b_h"])
        dW_emb = np.zeros((V, D), dtype=np.float32)

        dh_next = np.zeros((B, H), dtype=np.float32)
        dc_next = np.zeros((B, H), dtype=np.float32)

        for t in reversed(range(T)):
            dh_t = dhs[:, t, :] + dh_next
            c_t = cs[:, t, :]
            c_prev = cs[:, t - 1, :] if t > 0 else np.zeros((B, H), dtype=np.float32)
            h_prev = hs[:, t - 1, :] if t > 0 else np.zeros((B, H), dtype=np.float32)

            f_t = fs[:, t, :]
            i_t = iss[:, t, :]
            g_t = gs[:, t, :]
            o_t = os[:, t, :]
            c_tanh = c_t_tanhs[:, t, :]

            # Output gate gradient
            do_t = dh_t * c_tanh
            da_o = do_t * o_t * (1.0 - o_t)

            # Cell state gradient
            dc_t = dc_next + dh_t * o_t * (1.0 - np.square(c_tanh))

            # Candidate gate gradient
            dg_t = dc_t * i_t
            da_g = dg_t * (1.0 - np.square(g_t))

            # Input gate gradient
            di_t = dc_t * g_t
            da_i = di_t * i_t * (1.0 - i_t)

            # Forget gate gradient
            df_t = dc_t * c_prev
            da_f = df_t * f_t * (1.0 - f_t)

            # Propagate to previous cell state
            dc_next = dc_t * f_t

            # Concatenate gate pre-activations
            da_t = np.hstack([da_f, da_i, da_g, da_o])  # (B, 4H)

            # Accumulate weight gradients
            x_t = xs[:, t, :]  # (B, D)
            dW_xh += np.dot(x_t.T, da_t)
            dW_hh += np.dot(h_prev.T, da_t)
            db_h += np.sum(da_t, axis=0)

            # Gradient wrt inputs and previous hidden state
            dx_t = np.dot(da_t, W_xh.T)        # (B, D)
            dh_next = np.dot(da_t, W_hh.T)     # (B, H)

            # Accumulate embedding gradients
            token_ids = X[:, t]
            for b in range(B):
                t_id = token_ids[b]
                if t_id != 0:  # Skip PAD token updates
                    dW_emb[t_id] += dx_t[b]

        grads = {
            "W_emb": dW_emb,
            "W_xh": dW_xh,
            "W_hh": dW_hh,
            "b_h": db_h,
            "W_hy": dW_hy,
            "b_y": db_y
        }
        return grads

    def predict(self, X: np.ndarray, lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts top class index and confidence for input sequences.
        """
        probs, _ = self.forward(X, lengths)
        preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        return preds, confidences

    def save_weights(self, filepath: str):
        """Saves model weights and architecture hyperparameters into a single .npz file."""
        meta = np.array([
            self.vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            self.output_dim,
            self.seed
        ], dtype=np.int32)
        
        np.savez_compressed(
            filepath,
            meta=meta,
            W_emb=self.params["W_emb"],
            W_xh=self.params["W_xh"],
            W_hh=self.params["W_hh"],
            b_h=self.params["b_h"],
            W_hy=self.params["W_hy"],
            b_y=self.params["b_y"]
        )

    @classmethod
    def load_weights(cls, filepath: str) -> "NumPyLSTM":
        """Loads a model from a saved .npz file."""
        data = np.load(filepath)
        meta = data["meta"]
        vocab_size = int(meta[0])
        embedding_dim = int(meta[1])
        hidden_dim = int(meta[2])
        output_dim = int(meta[3])
        seed = int(meta[4])

        model = cls(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seed=seed
        )
        model.params["W_emb"] = data["W_emb"].astype(np.float32)
        model.params["W_xh"] = data["W_xh"].astype(np.float32)
        model.params["W_hh"] = data["W_hh"].astype(np.float32)
        model.params["b_h"] = data["b_h"].astype(np.float32)
        model.params["W_hy"] = data["W_hy"].astype(np.float32)
        model.params["b_y"] = data["b_y"].astype(np.float32)
        return model
