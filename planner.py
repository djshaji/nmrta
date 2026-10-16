"""
Prefrontal Active Inference Planner (Model Predictive Path Integral - MPPI).
Runs vectorized counterfactual rollouts in latent space across N candidates
in standard FP32 neural network precision, balancing extrinsic distance cost
with action regularization.
"""

import numpy as np

class PrefrontalMPPIPlanner:
    def __init__(self, world_model, horizon=20, num_rollouts=64, beta=0.05, seed=42):
        self.wm = world_model
        self.H = horizon
        self.N = num_rollouts
        self.beta = beta
        self.d_act = 2
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.lambda_param = 1.0
        
        # Pre-cast FP32 rollout matrices for vectorized MPPI imagination
        self.U_zr32 = np.hstack([self.wm.U_z, self.wm.U_r]).astype(np.float32)
        self.Uh32 = self.wm.U_h.astype(np.float32)
        self.W_zru32 = np.hstack([self.wm.W_z[-2:], self.wm.W_r[-2:]]).astype(np.float32)
        self.Whu32 = self.wm.W_h[-2:].astype(np.float32)
        self.W_dec32 = self.wm.W_dec[:self.wm.d_det].astype(np.float32)
        self.Wz_lat32 = self.wm.W_z[:-2].astype(np.float32)
        self.Wr_lat32 = self.wm.W_r[:-2].astype(np.float32)
        self.Wh_lat32 = self.wm.W_h[:-2].astype(np.float32)
        self.bz32 = self.wm.b_z.astype(np.float32)
        self.br32 = self.wm.b_r.astype(np.float32)
        self.bh32 = self.wm.b_h.astype(np.float32)

    def plan_action_chunk(self, h_current, z_current, target_pos, chunk_size=10):
        # Sample N candidate action sequences [N, H, d_act] in FP32
        delta_u = self.rng.normal(0.0, 0.4, (self.N, self.H, self.d_act)).astype(np.float32)
        
        # Batch tile initial deterministic recurrent state: [N, d_det]
        h_batch = np.tile(np.asarray(h_current, dtype=np.float32), (self.N, 1))
        zc = np.asarray(z_current, dtype=np.float32)
        
        # Precompute latent state contribution once before horizon rollout
        bzr = np.concatenate([zc @ self.Wz_lat32 + self.bz32, zc @ self.Wr_lat32 + self.br32])
        bh = zc @ self.Wh_lat32 + self.bh32
        targ = np.asarray(target_pos, dtype=np.float32)
        
        costs = np.zeros(self.N, dtype=np.float32)
        d_det = self.wm.d_det
        
        # Vectorized rollout across H time steps
        for t in range(self.H):
            u_t = np.clip(delta_u[:, t, :], -2.0, 2.0)
            
            # Fused FP32 GRU gate evaluation
            zr = 1.0 / (1.0 + np.exp(-(u_t @ self.W_zru32 + h_batch @ self.U_zr32 + bzr)))
            z_gate = zr[:, :d_det]
            r_gate = zr[:, d_det:]
            h_cand = np.tanh(u_t @ self.Whu32 + (r_gate * h_batch) @ self.Uh32 + bh)
            h_batch = (1.0 - z_gate) * h_batch + z_gate * h_cand
            
            # Readout predicted coordinates
            pred_ee = np.tanh(h_batch @ self.W_dec32) * 2.0
            
            # Batch extrinsic distance cost
            ext_costs = np.linalg.norm(pred_ee - targ, axis=-1)
            act_costs = 0.01 * np.sum(u_t**2, axis=-1)
            costs += ext_costs + act_costs
            
        min_cost = float(np.min(costs))
        weights = np.exp(-(costs - min_cost) / self.lambda_param)
        weights /= np.sum(weights)
        
        best_u = np.sum(weights[:, None, None] * delta_u, axis=0)
        action_chunk = best_u[:chunk_size].astype(np.float64)
        return action_chunk, min_cost
