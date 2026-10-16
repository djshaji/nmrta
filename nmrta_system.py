"""
Integrated NMRTA Cognitive Architecture.
Coordinates Thalamic Routing, Categorical World Modeling, Episodic Memory,
Active Inference Planning, and Cerebellar Sensorimotor Gating.
"""

import numpy as np
from environment import TwoDOFManipulator
from thalamus import ThalamicRouter
from world_model import CategoricalRSSM
from memory import ToroidalGridCellMap, ContinuousHopfieldMemory
from cerebellum import CerebellarForwardModel
from planner import PrefrontalMPPIPlanner

class NMRTASystem:
    def __init__(self, seed=42):
        self.seed = seed if seed is not None else 42
        self.rng = np.random.RandomState(self.seed)
        self.env = TwoDOFManipulator(seed_noise=self.seed)
        self.router = ThalamicRouter(seed=self.seed)
        self.wm = CategoricalRSSM(seed=self.seed)
        self.grid_map = ToroidalGridCellMap()
        self.memory = ContinuousHopfieldMemory(seed=self.seed)
        self.cerebellum = CerebellarForwardModel(seed=self.seed)
        self.planner = PrefrontalMPPIPlanner(self.wm, seed=self.seed)
        self.reset()

    def reset(self, target=None):
        self.obs = self.env.reset(target=target)
        self.h = np.zeros(self.wm.d_det)
        self.z = np.zeros(self.wm.d_lat)
        self.prior_probs = np.ones((self.wm.N_z, self.wm.K_z)) / self.wm.K_z
        self.current_time = 0.0
        self.cerebellum.reset()
        
        self.history = {
            "time": [],
            "ee_pos": [],
            "target": [],
            "efference_error": [],
            "reflex_tripped": [],
            "damping_gain": [],
            "action_torques": [],
            "attention_allocation": [],
            "kl_surprise": [],
            "hopfield_writes": []
        }

    def train_cerebellar_baseline(self, num_steps=2000):
        s = self.env.reset()
        rng = np.random.RandomState(self.seed + 1000)
        X, Y = [], []
        for _ in range(num_steps):
            u = rng.uniform(-5.0, 5.0, 2)
            s_next, _, _, _ = self.env.step(u)
            delta_s = s_next - s
            X.append(np.concatenate([s, u]))
            Y.append(delta_s)
            s = s_next
        self.cerebellum.fit_from_data(np.array(X), np.array(Y))

    def run_trial(self, duration=10.0, perturbation_schedule=None, compensation_enabled=True):
        """
        Executes a multi-rate cognitive-sensorimotor trial.
        duration: total trial duration in seconds (default 10.0 s = 100 cognitive steps = 1000 motor steps)
        perturbation_schedule: list of tuples (t_start, t_end, force_vector_2d)
        compensation_enabled: True for nominal efference copy cancellation, False for ablation
        """
        total_cog_steps = int(round(duration / 0.100))
        if perturbation_schedule is None:
            perturbation_schedule = []
            
        for cog_step in range(total_cog_steps):
            v, a, p = self.env.get_multimodal_tokens()
            E_all = self.router.project_modalities(v, a, p)
            e_t, bar_e_t, attn_weights = self.router.forward(E_all)
            attn_dist = self.router.get_modality_attention_distribution(attn_weights)
            
            z_post, post_probs = self.wm.posterior(self.h, bar_e_t)
            kl_surprise = self.wm.kl_divergence(post_probs, self.prior_probs)
            self.z = z_post
            
            ee_pos = self.env.forward_kinematics()
            grid_feat = self.grid_map.encode(ee_pos)
            hop_written = self.memory.write(self.h, grid_feat, kl_surprise)
            
            action_chunk, plan_cost = self.planner.plan_action_chunk(
                self.h, self.z, self.env.target, chunk_size=10
            )
            
            u_executed = np.zeros(2)
            for k in range(10):
                t_now = round(self.env.time, 4)
                u_cmd = action_chunk[k]
                u_executed = u_cmd
                s_prev = self.env.get_proprioceptive_state()
                
                # Check active external perturbation
                f_ext = np.zeros(2, dtype=np.float64)
                for (p_start, p_end, p_force) in perturbation_schedule:
                    if p_start <= t_now < p_end:
                        f_ext = np.array(p_force, dtype=np.float64)
                        break
                        
                obs, reward, done, info = self.env.step(u_cmd, external_force=f_ext)
                s_actual = obs
                
                eps, reflex_tripped, damping = self.cerebellum.step_efference_check(
                    s_prev, u_cmd, s_actual, t_now, compensation_enabled=compensation_enabled
                )
                
                # If reflex tripped, apply reflex damping stabilization
                if reflex_tripped and damping > 0:
                    self.env.step([0.0, 0.0], damping_gain=damping)
                    
                self.history["time"].append(t_now)
                self.history["ee_pos"].append(info["ee_pos"].copy())
                self.history["target"].append(info["target"].copy())
                self.history["efference_error"].append(eps)
                self.history["reflex_tripped"].append(reflex_tripped)
                self.history["damping_gain"].append(damping)
                self.history["action_torques"].append(u_cmd.copy())
                self.history["attention_allocation"].append(attn_dist)
                self.history["kl_surprise"].append(kl_surprise)
                self.history["hopfield_writes"].append(hop_written)
                
            gru_in = np.concatenate([self.z, u_executed])
            self.h = self.wm.gru_cell(gru_in, self.h)
            self.prior_z, self.prior_probs = self.wm.prior(self.h)
