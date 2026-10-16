"""
Categorical Recurrent State-Space Model (RSSM) Core.
Provides discrete categorical latent states, recurrent GRU dynamics,
and predictive rollouts with independent instance RNG.
"""

import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class CategoricalRSSM:
    def __init__(self, d_det=256, N_z=16, K_z=16, d_act=2, d_obs=512, seed=42):
        self.d_det = d_det
        self.N_z = N_z
        self.K_z = K_z
        self.d_lat = N_z * K_z
        self.d_act = d_act
        self.d_obs = d_obs
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        d_in = self.d_lat + d_act
        self.W_z = self.rng.randn(d_in, d_det) * np.sqrt(2.0 / d_in)
        self.U_z = self.rng.randn(d_det, d_det) * np.sqrt(2.0 / d_det)
        self.b_z = np.zeros(d_det)
        
        self.W_r = self.rng.randn(d_in, d_det) * np.sqrt(2.0 / d_in)
        self.U_r = self.rng.randn(d_det, d_det) * np.sqrt(2.0 / d_det)
        self.b_r = np.zeros(d_det)
        
        self.W_h = self.rng.randn(d_in, d_det) * np.sqrt(2.0 / d_in)
        self.U_h = self.rng.randn(d_det, d_det) * np.sqrt(2.0 / d_det)
        self.b_h = np.zeros(d_det)
        
        self.W_prior1 = self.rng.randn(d_det, 128) * np.sqrt(2.0 / d_det)
        self.W_prior2 = self.rng.randn(128, self.d_lat) * np.sqrt(2.0 / 128)
        
        d_post_in = d_det + d_obs
        self.W_post1 = self.rng.randn(d_post_in, 128) * np.sqrt(2.0 / d_post_in)
        self.W_post2 = self.rng.randn(128, self.d_lat) * np.sqrt(2.0 / 128)
        
        self.W_dec = self.rng.randn(d_det + self.d_lat, 2) * np.sqrt(2.0 / (d_det + self.d_lat))

    def gru_cell(self, x, h):
        z_gate = 1.0 / (1.0 + np.exp(-(x @ self.W_z + h @ self.U_z + self.b_z)))
        r_gate = 1.0 / (1.0 + np.exp(-(x @ self.W_r + h @ self.U_r + self.b_r)))
        h_cand = np.tanh(x @ self.W_h + (r_gate * h) @ self.U_h + self.b_h)
        return (1.0 - z_gate) * h + z_gate * h_cand

    def sample_categorical(self, logits):
        probs = softmax(logits, axis=-1)
        shape = probs.shape
        flat_probs = probs.reshape(-1, self.K_z)
        samples = np.zeros_like(flat_probs)
        for i in range(flat_probs.shape[0]):
            chosen = self.rng.choice(self.K_z, p=flat_probs[i])
            samples[i, chosen] = 1.0
        return samples.reshape(shape), probs

    def prior(self, h):
        hid = np.maximum(0, h @ self.W_prior1)
        logits = (hid @ self.W_prior2).reshape(self.N_z, self.K_z)
        sample, probs = self.sample_categorical(logits)
        return sample.flatten(), probs

    def posterior(self, h, bar_e):
        inp = np.concatenate([h, bar_e])
        hid = np.maximum(0, inp @ self.W_post1)
        logits = (hid @ self.W_post2).reshape(self.N_z, self.K_z)
        sample, probs = self.sample_categorical(logits)
        return sample.flatten(), probs

    def kl_divergence(self, post_probs, prior_probs, eps=1e-7):
        p = np.clip(prior_probs, eps, 1.0)
        q = np.clip(post_probs, eps, 1.0)
        return float(np.sum(q * np.log(q / p)))

    def imagine_step(self, h, z, action):
        gru_in = np.concatenate([z, action])
        h_next = self.gru_cell(gru_in, h)
        z_next, probs = self.prior(h_next)
        pred_ee = np.tanh(np.concatenate([h_next, z_next]) @ self.W_dec) * 2.0
        return h_next, z_next, probs, pred_ee

    def imagine_rollout(self, h0, z0, action_sequence):
        h = h0.copy()
        z = z0.copy()
        pred_coords = []
        prior_probs_list = []
        for a in action_sequence:
            h, z, probs, ee = self.imagine_step(h, z, a)
            pred_coords.append(ee)
            prior_probs_list.append(probs)
        return np.array(pred_coords), prior_probs_list
