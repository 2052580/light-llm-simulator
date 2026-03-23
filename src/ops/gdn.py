from src.ops.base import BaseOp


class OpGDNLinear(BaseOp):
    PHI_Q_FACTOR = 2.0
    PHI_K_FACTOR = 2.0

    def __init__(self, config, elem_size=2):
        super().__init__("GDNLinear", config.aichip_config, elem_size)
        self.config = config
        self.model_config = config.model_config
        self.attn_bs = config.attn_bs
        self.seq_len = config.seq_len

    def compute_cost(self):
        hk = getattr(self.model_config, "linear_num_key_heads", 0)
        dk = getattr(self.model_config, "linear_key_head_dim", 0)
        hv = getattr(self.model_config, "linear_num_value_heads", 0)
        dv = getattr(self.model_config, "linear_value_head_dim", 0)
        tp = max(1, getattr(self.config, "attn_tensor_parallel", 1))
        b = self.attn_bs
        s = self.seq_len
        phi_q = self.PHI_Q_FACTOR * b * s * (hk * dk / tp)
        phi_k = self.PHI_K_FACTOR * b * s * (hk * dk / tp)
        state_update = 2 * b * s * (hk * dk * dv / tp)
        readout = 2 * b * s * (hk * dk * dv / tp)
        self.compute_flops = phi_q + phi_k + state_update + readout
        phi_time = (phi_q + phi_k) / self.vec_flops_fp16
        sr_time = (state_update + readout) / self.cube_flops_fp16
        self.compute_time = phi_time + sr_time
        return self.compute_time

    def memory_cost(self):
        hk = getattr(self.model_config, "linear_num_key_heads", 0)
        dk = getattr(self.model_config, "linear_key_head_dim", 0)
        hv = getattr(self.model_config, "linear_num_value_heads", 0)
        dv = getattr(self.model_config, "linear_value_head_dim", 0)
        tp = max(1, getattr(self.config, "attn_tensor_parallel", 1))
        b = self.attn_bs
        s = self.seq_len
        state_size = hk * dk * dv / tp
        bytes_state = 2 * self.elem_size * b * s * state_size
        bytes_feats = self.elem_size * b * s * ((hk * dk + hv * dv) / tp)
        bytes_out = self.elem_size * b * s * (hk * dv / tp)
        self.bytes = bytes_state + bytes_feats + bytes_out
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time
