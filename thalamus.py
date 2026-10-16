"""
Thalamic Routing Engine (Global Workspace Bottleneck).
Implements multimodal projection and Perceiver IO cross-attention with M=64 latent query tokens,
token-mean pooling, cross-modal saliency gating, and an unbottlenecked 352-token pathway for ablation.
"""

import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

class ThalamicRouter:
    def __init__(self, d=512, M=64, heads=8, seed=42):
        self.d = d
        self.M = M
        self.heads = heads
        self.d_k = d // heads
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.W_proj_v = self.rng.randn(384, d) * np.sqrt(2.0 / 384)
        self.W_proj_a = self.rng.randn(256, d) * np.sqrt(2.0 / 256)
        self.W_proj_p = self.rng.randn(128, d) * np.sqrt(2.0 / 128)
        
        self.L = self.rng.randn(M, d) * 0.02
        
        self.W_q = self.rng.randn(d, d) * np.sqrt(1.0 / d)
        self.W_k = self.rng.randn(d, d) * np.sqrt(1.0 / d)
        self.W_v = self.rng.randn(d, d) * np.sqrt(1.0 / d)
        self.W_out = self.rng.randn(d, d) * np.sqrt(1.0 / d)
        
        self.W_mlp1 = self.rng.randn(d, 2 * d) * np.sqrt(1.0 / d)
        self.W_mlp2 = self.rng.randn(2 * d, d) * np.sqrt(1.0 / (2 * d))
        
        # Precomputed associative projection operators (exact to machine precision)
        self.Q = self.L @ self.W_q
        self.QWk = (self.Q @ self.W_k.T) / np.sqrt(self.d)
        self.W_vo = self.W_v @ self.W_out

    def project_modalities(self, E_v, E_a, E_p):
        tilde_E_v = E_v @ self.W_proj_v
        tilde_E_a = E_a @ self.W_proj_a
        tilde_E_p = E_p @ self.W_proj_p
        return np.vstack([tilde_E_v, tilde_E_a, tilde_E_p])

    def forward(self, E_all):
        """
        64-token Global Workspace Bottleneck cross-attention pass.
        Compresses E_all [352, d] into e_t [64, d] and pooled vector bar_e_t [d].
        """
        scores = self.QWk @ E_all.T
        attn_weights = softmax(scores, axis=-1)
        
        attn_out = (attn_weights @ E_all) @ self.W_vo
        lat = layer_norm(self.L + attn_out)
        
        mlp_h = lat @ self.W_mlp1
        gelu_h = 0.5 * mlp_h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (mlp_h + 0.044715 * (mlp_h**3))))
        mlp_out = gelu_h @ self.W_mlp2
        
        e_t = layer_norm(lat + mlp_out)
        bar_e_t = np.mean(e_t, axis=0)
        return e_t, bar_e_t, attn_weights

    def forward_unbottlenecked(self, E_all):
        """
        Unbottlenecked 352-token pathway (bypasses the M=64 Perceiver IO bottleneck).
        Pools across all 352 multimodal tokens with residual value-output projection.
        """
        mean_E = np.mean(E_all, axis=0)
        bar_e_352 = layer_norm(mean_E + (mean_E @ self.W_vo))
        return E_all, bar_e_352

    def get_modality_attention_distribution(self, attn_weights):
        mean_attn = np.mean(attn_weights, axis=0)
        v_attn = np.sum(mean_attn[:256])
        a_attn = np.sum(mean_attn[256:320])
        p_attn = np.sum(mean_attn[320:352])
        total = v_attn + a_attn + p_attn
        return {
            "visual": float(v_attn / total),
            "auditory": float(a_attn / total),
            "somatosensory": float(p_attn / total)
        }
