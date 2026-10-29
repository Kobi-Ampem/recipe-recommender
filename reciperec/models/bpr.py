"""BPR matrix factorisation (Rendle et al., 2009) in TensorFlow.

Same shape as PureSVD: every user and recipe gets an embedding of size
`factors`, and score(u, i) = user_u . item_i + bias_i.

The difference is what training asks for. PureSVD treats every unreviewed
recipe as a 0, as if the user disliked it. BPR only asks that a recipe the
user reviewed (positive) scores above a random one they did not (negative):
loss = -log sigmoid(score(u, pos) - score(u, neg)). That targets ranking
directly, which is what the metrics measure.

Negatives are sampled uniformly from the whole catalogue without checking
the user's history. With Food.com density under 0.1%, a sampled
"negative" is almost never one of the user's own recipes.

With a validation matrix, fit() scores it every `eval_every` epochs, keeps
the best weights and stops after `patience` checks without improvement.
The best epoch is recorded so the final test run can train for exactly
that long without looking at test.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from reciperec.metrics import evaluate


class BPR:
    def __init__(
        self,
        factors: int = 64,
        lr: float = 0.01,
        reg: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 4096,
        patience: int = 3,
        eval_every: int = 2,
        seed: int = 0,
    ):
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.eval_every = eval_every
        self.seed = seed
        self.fit_info: dict = {}

    def _build(self, n_users: int, n_items: int) -> None:
        tf.random.set_seed(self.seed)
        init = tf.keras.initializers.RandomNormal(stddev=0.01, seed=self.seed)
        self.user = tf.Variable(init((n_users, self.factors)), name="user")
        self.item = tf.Variable(init((n_items, self.factors)), name="item")
        self.bias = tf.Variable(tf.zeros(n_items), name="bias")
        self.opt = tf.keras.optimizers.Adam(self.lr)
        variables = [self.user, self.item, self.bias]

        @tf.function
        def step(u, pos, neg):
            with tf.GradientTape() as tape:
                eu = tf.gather(self.user, u)
                ep, en = tf.gather(self.item, pos), tf.gather(self.item, neg)
                diff = tf.reduce_sum(eu * (ep - en), axis=1)
                diff += tf.gather(self.bias, pos) - tf.gather(self.bias, neg)
                loss = -tf.reduce_mean(tf.math.log_sigmoid(diff))
                penalty = tf.reduce_mean(tf.reduce_sum(eu**2 + ep**2 + en**2, axis=1))
                total = loss + self.reg * penalty
            grads = tape.gradient(total, variables)
            self.opt.apply_gradients(zip(grads, variables))
            return loss

        self._step = step

    def _weights(self) -> list[np.ndarray]:
        return [v.numpy() for v in (self.user, self.item, self.bias)]

    def fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None = None) -> BPR:
        rng = np.random.default_rng(self.seed)
        n_users, n_items = train.shape
        self._build(n_users, n_items)
        coo = train.tocoo()
        users, items = coo.row.astype(np.int32), coo.col.astype(np.int32)

        best, best_epoch, best_weights, bad = -1.0, self.epochs, None, 0
        history = []
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(len(users))
            negs = rng.integers(0, n_items, len(users), dtype=np.int32)
            losses = []
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                losses.append(float(self._step(users[idx], items[idx], negs[idx])))
            if val is None or epoch % self.eval_every:
                continue
            ndcg = evaluate(self, train, val)["ndcg@10"]
            history.append({"epoch": epoch, "loss": round(np.mean(losses), 4), "ndcg@10": ndcg})
            print(f"    epoch {epoch:>3}  loss {np.mean(losses):.4f}  val ndcg@10 {ndcg:.4f}")
            if ndcg > best:
                best, best_epoch, best_weights, bad = ndcg, epoch, self._weights(), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_weights is not None:
            for var, w in zip((self.user, self.item, self.bias), best_weights):
                var.assign(w)
            self.fit_info = {"best_epoch": best_epoch, "history": history}
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        u = tf.gather(self.user, users)
        return (tf.matmul(u, self.item, transpose_b=True) + self.bias).numpy()
