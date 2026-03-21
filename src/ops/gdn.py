from src.ops.base import BaseOp


class OpGDNLinear(BaseOp):
    def __init__(self, config, elem_size=2):
        super().__init__("GDNLinear", config.aichip_config, elem_size)
        self.config = config
        self.model_config = config.model_config
        self.attn_bs = config.attn_bs
        self.seq_len = config.seq_len

    def compute_cost(self):
        k_heads = getattr(self.model_config, "linear_num_key_heads", 0)
        k_dim = getattr(self.model_config, "linear_key_head_dim", 0)
        v_heads = getattr(self.model_config, "linear_num_value_heads", 0)
        v_dim = getattr(self.model_config, "linear_value_head_dim", 0)
        proj_k = 2 * self.attn_bs * self.seq_len * k_heads * k_dim
        proj_v = 2 * self.attn_bs * self.seq_len * v_heads * v_dim
        mix = 2 * self.attn_bs * self.seq_len * k_dim * v_dim
        self.compute_flops = proj_k + proj_v + mix
        self.compute_time = self.compute_flops / self.vector_flops
        return self.compute_time

    def memory_cost(self):
        k_heads = getattr(self.model_config, "linear_num_key_heads", 0)
        k_dim = getattr(self.model_config, "linear_key_head_dim", 0)
        v_heads = getattr(self.model_config, "linear_num_value_heads", 0)
        v_dim = getattr(self.model_config, "linear_value_head_dim", 0)
        self.bytes = self.elem_size * self.attn_bs * self.seq_len * (k_heads * k_dim + v_heads * v_dim)
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time
