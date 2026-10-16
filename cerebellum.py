"""
100 Hz Cerebellar Forward Model & Efference Copy Gating.
Predicts immediate sensory increment via a 74-dimensional nonlinear Fourier-kinematic
basis over joint configuration and motor command, cancels self-generated sensations,
detects external force perturbations via reflex threshold tau_refl = 0.12 rad/s,
and applies refractory hysteresis and damping backoff to prevent plan thrashing.
Includes native support for ablation condition (compensation_enabled=False).
"""

import math
import numpy as np

class CerebellarForwardModel:
    def __init__(self, d_proprio=4, d_act=2, tau_refl=0.12, seed=42):
        self.d_proprio = d_proprio
        self.d_act = d_act
        self.tau_refl = float(tau_refl)  # 0.12 rad/s
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # 74-dimensional nonlinear Fourier-kinematic basis weights
        self.d_feat = 74
        self.W = np.zeros((self.d_feat, d_proprio), dtype=np.float64)
        
        # Anti-thrashing parameters
        self.refractory_window = 0.150  # 150 ms refractory hold
        self.last_interrupt_time = -1.0
        self.interrupt_count_window = []
        self.base_damping = 80.0  # K_{d, refl} = 80.0 N*m*s/rad

    @staticmethod
    def extract_features(s, u):
        """
        Maps proprioceptive state s = [q1, q2, dq1, dq2] and action u = [u1, u2]
        to a 74-dimensional tensor-product Fourier-kinematic feature vector
        capturing configuration-dependent inverse inertia and Coriolis terms.
        """
        q1, q2, dq1, dq2 = float(s[0]), float(s[1]), float(s[2]), float(s[3])
        u1, u2 = float(u[0]), float(u[1])
        c1, s1 = math.cos(q2), math.sin(q2)
        c2, s2 = math.cos(2.0 * q2), math.sin(2.0 * q2)
        c3, s3 = math.cos(3.0 * q2), math.sin(3.0 * q2)
        c4, s4 = math.cos(4.0 * q2), math.sin(4.0 * q2)
        
        dq1_sq = dq1 * dq1
        dq2_sq = dq2 * dq2
        dq12 = dq1 * dq2
        
        b = (1.0, c1, s1, c2, s2, c3, s3, c4, s4)
        k = (1.0, u1, u2, dq1, dq2, dq1_sq, dq2_sq, dq12)
        out = [q1, q2]
        for kv in k:
            for bv in b:
                out.append(kv * bv)
        return np.array(out, dtype=np.float64)

    def fit_from_data(self, X, Y, ridge_lambda=1e-3):
        """
        Fits cerebellar forward dynamics from voluntary transition data.
        X: [N, d_proprio + d_act] raw transitions or [N, 74] pre-extracted features
        Y: [N, d_proprio] (delta_s)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[1] == self.d_proprio + self.d_act:
            Phi = np.array([self.extract_features(row[:4], row[4:]) for row in X_arr], dtype=np.float64)
        else:
            Phi = X_arr
        I = np.eye(Phi.shape[1], dtype=np.float64)
        self.W = np.linalg.solve(Phi.T @ Phi + ridge_lambda * I, Phi.T @ Y)

    def predict_delta_s(self, s, u):
        phi = self.extract_features(s, u)
        return phi @ self.W

    def step_efference_check(self, s_prev, u, s_actual, current_time, compensation_enabled=True):
        """
        Computes efference copy prediction error:
        If compensation_enabled=True:
            epsilon_{t+k} = || s_actual[2:] - (s_prev[2:] + delta_s_hat[2:]) ||_2 (on velocities)
        If compensation_enabled=False (ablation):
            epsilon = || s_actual[2:] - s_prev[2:] ||_2 (raw uncompensated velocity delta)
        """
        if compensation_enabled:
            delta_s_hat = self.predict_delta_s(s_prev, u)
            s_pred = s_prev + delta_s_hat
            vel_err = s_actual[2:] - s_pred[2:]
        else:
            vel_err = s_actual[2:] - s_prev[2:]
            
        epsilon = float(np.linalg.norm(vel_err))
        
        reflex_tripped = False
        damping_gain = 0.0
        
        if epsilon >= self.tau_refl:
            if (current_time - self.last_interrupt_time) >= self.refractory_window:
                reflex_tripped = True
                self.last_interrupt_time = current_time
                self.interrupt_count_window.append(current_time)
                
            self.interrupt_count_window = [t for t in self.interrupt_count_window if current_time - t <= 0.200]
            if len(self.interrupt_count_window) >= 3:
                damping_gain = self.base_damping * 1.5
            else:
                damping_gain = self.base_damping
                
        return epsilon, reflex_tripped, damping_gain

    def reset(self):
        self.last_interrupt_time = -1.0
        self.interrupt_count_window = []
