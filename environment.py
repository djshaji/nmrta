"""
2-DOF Planar Robotic Manipulator Diagnostic Simulation Environment.
Provides realistic continuous kinematics, nonlinear dynamics, sensor noise,
and physical perturbation force injection with independent instance RNG.
"""

import numpy as np

class TwoDOFManipulator:
    def __init__(self, l1=1.0, l2=0.8, m1=0.25, m2=0.20, dt=0.01,
                 pos_noise_std=0.005, vel_noise_std=0.010, seed_noise=42):
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.dt = float(dt)  # 100 Hz sensorimotor step (10 ms)
        self.pos_noise_std = float(pos_noise_std)  # 0.005 rad
        self.vel_noise_std = float(vel_noise_std)  # 0.010 rad/s
        self.seed_noise = seed_noise
        self.rng = np.random.RandomState(seed_noise)
        self.reset()

    def reset(self, q0=None, qdot0=None, target=None):
        if q0 is None:
            self.q = np.array([0.5, 0.5], dtype=np.float64)
        else:
            self.q = np.array(q0, dtype=np.float64)
            
        if qdot0 is None:
            self.qdot = np.zeros(2, dtype=np.float64)
        else:
            self.qdot = np.array(qdot0, dtype=np.float64)
            
        if target is None:
            self.target = np.array([1.2, 0.8], dtype=np.float64)
        else:
            self.target = np.array(target, dtype=np.float64)
            
        self.time = 0.0
        return self.get_proprioceptive_state()

    def forward_kinematics(self, q=None):
        if q is None:
            q = self.q
        t1, t2 = q[0], q[1]
        x1 = self.l1 * np.cos(t1)
        y1 = self.l1 * np.sin(t1)
        x2 = x1 + self.l2 * np.cos(t1 + t2)
        y2 = y1 + self.l2 * np.sin(t1 + t2)
        return np.array([x2, y2], dtype=np.float64)

    def jacobian(self, q=None):
        if q is None:
            q = self.q
        t1, t2 = q[0], q[1]
        j11 = -self.l1 * np.sin(t1) - self.l2 * np.sin(t1 + t2)
        j12 = -self.l2 * np.sin(t1 + t2)
        j21 = self.l1 * np.cos(t1) + self.l2 * np.cos(t1 + t2)
        j22 = self.l2 * np.cos(t1 + t2)
        return np.array([[j11, j12], [j21, j22]], dtype=np.float64)

    def mass_matrix(self, q=None):
        if q is None:
            q = self.q
        t2 = q[1]
        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        
        m11 = (m1 + m2) * (l1**2) + m2 * (l2**2) + 2.0 * m2 * l1 * l2 * np.cos(t2)
        m12 = m2 * (l2**2) + m2 * l1 * l2 * np.cos(t2)
        m21 = m12
        m22 = m2 * (l2**2)
        return np.array([[m11, m12], [m21, m22]], dtype=np.float64)

    def coriolis_gravity(self, q=None, qdot=None):
        if q is None:
            q = self.q
        if qdot is None:
            qdot = self.qdot
        t1, t2 = q[0], q[1]
        d1, d2 = qdot[0], qdot[1]
        l1, l2 = self.l1, self.l2
        m2 = self.m2
        
        h = -m2 * l1 * l2 * np.sin(t2)
        c1 = h * (2.0 * d1 * d2 + d2**2)
        c2 = -h * (d1**2)
        
        b = 0.5  # viscous joint damping (N*m*s/rad)
        f1 = b * d1
        f2 = b * d2
        return np.array([c1 + f1, c2 + f2], dtype=np.float64)

    def step(self, action_torques, external_force=None, damping_gain=0.0):
        u = np.array(action_torques, dtype=np.float64)
        u = np.clip(u, -50.0, 50.0)  # Torque limits [-50, 50] N*m
        
        if damping_gain > 0.0:
            u -= damping_gain * self.qdot
            
        tau_ext = np.zeros(2, dtype=np.float64)
        if external_force is not None and np.linalg.norm(external_force) > 0:
            J = self.jacobian(self.q)
            tau_ext = J.T @ np.array(external_force, dtype=np.float64)
            
        net_torque = u + tau_ext
        M = self.mass_matrix(self.q)
        C = self.coriolis_gravity(self.q, self.qdot)
        qddot = np.linalg.solve(M, net_torque - C)
        
        # Sub-stepped integration (10 sub-steps of 1 ms)
        n_sub = 10
        dt_sub = self.dt / n_sub
        for _ in range(n_sub):
            self.qdot += qddot * dt_sub
            self.q += self.qdot * dt_sub
            
        # Joint limits [-pi, pi] rad
        self.q = np.clip(self.q, -np.pi, np.pi)
        self.time += self.dt
        
        ee_pos = self.forward_kinematics(self.q)
        dist_to_target = np.linalg.norm(ee_pos - self.target)
        
        obs = self.get_proprioceptive_state()
        reward = -float(dist_to_target)
        done = dist_to_target < 0.05 or self.time > 10.0
        
        info = {
            "ee_pos": ee_pos,
            "target": self.target,
            "dist_to_target": dist_to_target,
            "qddot": qddot,
            "net_torque": net_torque
        }
        return obs, reward, done, info

    def get_proprioceptive_state(self):
        # Independent Gaussian noise on position (0.005 rad) and velocity (0.010 rad/s)
        q_noise = self.rng.normal(0.0, self.pos_noise_std, size=2)
        qdot_noise = self.rng.normal(0.0, self.vel_noise_std, size=2)
        return np.concatenate([self.q + q_noise, self.qdot + qdot_noise])

    def get_multimodal_tokens(self, d=512):
        ee_pos = self.forward_kinematics()
        target_vec = self.target - ee_pos
        
        # Multimodal sensory token generator with instance RNG
        v_base = np.tile(np.concatenate([ee_pos, self.target, target_vec]), 384 // 6)[:384]
        v_tokens = np.tile(v_base, (256, 1)) + self.rng.normal(0.0, 0.05, (256, 384))
        
        a_base = np.sin(np.linspace(0, 2 * np.pi, 256)) * (1.0 / (1.0 + np.linalg.norm(target_vec)))
        a_tokens = np.tile(a_base, (64, 1)) + self.rng.normal(0.0, 0.02, (64, 256))
        
        p_base = np.tile(np.concatenate([self.q, self.qdot]), 128 // 4)[:128]
        p_tokens = np.tile(p_base, (32, 1)) + self.rng.normal(0.0, 0.01, (32, 128))
        
        return v_tokens, a_tokens, p_tokens
