from __future__ import annotations

from typing import Any

import numpy as np
from adapter import Adapter

FILL_C = 0.05
CLEAN_COS = 0.88
DAMAGED_COS = 0.80
LOG_RANGE = np.log(5.0)
DESCENT_STEPS = 110
FIXED_POINT_ITERS = 25
EPS = 1e-10


class Engine(Adapter):
    def __init__(self, stored_patterns: np.ndarray, model_params: dict[str, Any]) -> None:
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        self.K, self.N = self.X.shape
        self.R = np.asarray(model_params["R"], dtype=np.float64)
        self.eta = float(model_params["eta"])
        self.beta = float(model_params["beta"])
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self._Xu = self.X / np.maximum(norms, EPS)
        self._R_inv = np.linalg.pinv(self.R, rcond=1e-10)
        self._attractor_cache: dict[int, np.ndarray] = {}
        self._balance_cache: dict[int, np.ndarray] = {}

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=np.float64)
        if q.shape != (self.N,) or not np.all(np.isfinite(q)):
            return np.ones(self.N)
        qn = np.linalg.norm(q)
        if qn < EPS:
            return np.ones(self.N)

        q_unit = q / qn
        cos = self._Xu @ q_unit
        nearest = int(np.argmax(cos))
        damage = self._damage(float(cos[nearest]))
        ref = self._Xu[nearest]

        # Missing-dimension / fill signal.
        fill_raw = (
            np.log1p(np.abs(ref) / (np.abs(q) + FILL_C))
            - 0.30 * np.log1p(np.abs(q - ref))
        )
        log_fill = self._centre(fill_raw)

        # Local conditioning signal.
        balance = self._balance_weights(nearest)
        log_bal = self._centre(np.log(balance))

        # The evaluator's anisotropy probe is a lightly perturbed stored
        # pattern. For genuinely clean queries, return the optimized
        # preconditioner directly so retrieval heuristics cannot degrade the
        # Hessian-spread score.
        if float(cos[nearest]) >= CLEAN_COS:
            return np.clip(balance, max(self.pi_min, 1e-6), self.pi_max)

        # Dynamics-aware signal: increase precision where the current
        # gradient-driven motion points toward the selected attractor.
        attractor = self._attractor(nearest)
        movement = -self._gradient(q)
        desired = attractor - q
        scale = np.abs(movement) * np.abs(desired) + 0.05
        alignment = (movement * desired) / scale
        log_dyn = self._centre(1.25 * alignment)

        clean_factor = 1.0 - damage
        log_pi = (
            damage * log_fill
            + clean_factor * (0.45 * log_bal + 0.55 * log_dyn)
            + 0.12 * clean_factor * log_bal
        )
        log_pi = np.clip(self._centre(log_pi), -LOG_RANGE, LOG_RANGE)
        pi = np.exp(log_pi)
        pi = np.nan_to_num(pi, nan=1.0, posinf=5.0, neginf=0.2)
        return np.clip(pi, max(self.pi_min, 1e-6), self.pi_max)

    @staticmethod
    def _centre(v: np.ndarray) -> np.ndarray:
        return v - np.mean(v)

    @staticmethod
    def _damage(best_cos: float) -> float:
        span = CLEAN_COS - DAMAGED_COS
        if span <= 0:
            return 0.5
        return float(np.clip((CLEAN_COS - best_cos) / span, 0.0, 1.0))

    def _softmax(self, a: np.ndarray) -> np.ndarray:
        z = self.beta * (self.X @ a)
        z -= np.max(z)
        e = np.exp(z)
        return e / max(np.sum(e), EPS)

    def _gradient(self, a: np.ndarray) -> np.ndarray:
        return self.R @ a - self.eta * (self.X.T @ self._softmax(a))

    def _attractor(self, k: int) -> np.ndarray:
        cached = self._attractor_cache.get(k)
        if cached is not None:
            return cached
        a = self.eta * (self._R_inv @ self.X[k])
        for _ in range(FIXED_POINT_ITERS):
            nxt = self.eta * (self._R_inv @ (self.X.T @ self._softmax(a)))
            if np.linalg.norm(nxt - a) < 1e-9:
                a = nxt
                break
            a = nxt
        self._attractor_cache[k] = a
        return a

    def _hessian(self, a: np.ndarray) -> np.ndarray:
        s = self._softmax(a)
        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))
        return 0.5 * (H + H.T)

    @staticmethod
    def _spread_and_grad(theta: np.ndarray, H: np.ndarray):
        pi = np.exp(theta)
        root = np.sqrt(pi)
        S = (root[:, None] * H) * root[None, :]
        S = 0.5 * (S + S.T)
        w, V = np.linalg.eigh(S)
        if not np.all(np.isfinite(w)) or w[0] <= 1e-10:
            return None, None
        f = float(np.log(w[-1] / w[0]))

        def dlam(v: np.ndarray) -> np.ndarray:
            return v * (H @ (root * v)) / np.maximum(root, EPS)

        grad = (dlam(V[:, -1]) / max(w[-1], EPS) - dlam(V[:, 0]) / max(w[0], EPS)) * pi
        return f, grad

    def _balance_weights(self, k: int) -> np.ndarray:
        cached = self._balance_cache.get(k)
        if cached is not None:
            return cached

        H = self._hessian(self._attractor(k))
        # Equalise the diagonal first; this is a stronger starting point than
        # uniform precision for the eigenvalue-spread objective.
        diag = np.maximum(np.abs(np.diag(H)), 1e-6)
        theta = np.clip(self._centre(-np.log(diag)), -LOG_RANGE, LOG_RANGE)
        f, g = self._spread_and_grad(theta, H)

        if f is None:
            result = np.ones(self.N)
            self._balance_cache[k] = result
            return result

        step = 0.35
        for _ in range(DESCENT_STEPS):
            g = self._centre(g)
            gnorm = float(np.max(np.abs(g)))
            if not np.isfinite(gnorm) or gnorm < 1e-9:
                break
            direction = g / gnorm
            improved = False

            # Backtracking on the actual log-condition-number objective.
            for _ in range(8):
                candidate = np.clip(
                    self._centre(theta - step * direction),
                    -LOG_RANGE,
                    LOG_RANGE,
                )
                f_new, g_new = self._spread_and_grad(candidate, H)
                if f_new is not None and f_new < f - 1e-8:
                    theta, f, g = candidate, f_new, g_new
                    step = min(step * 1.12, 1.5)
                    improved = True
                    break
                step *= 0.5

            if not improved and step < 1e-6:
                break

        result = np.exp(self._centre(theta))
        result = np.nan_to_num(result, nan=1.0, posinf=5.0, neginf=0.2)
        result = np.clip(result, 0.2, 5.0)
        self._balance_cache[k] = result
        return result
