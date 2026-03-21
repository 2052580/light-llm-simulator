from src.ops.base import BaseOp


class MLAFlashAttentionFP16(BaseOp):
    '''
    Description:
        The Flash Attention operation for the model used MLA attention mechanism in FP16 precision.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config, elem_size=2):
        self.op_disc_factor()
        super().__init__("MLAFlashAttentionFP16", config.aichip_config, elem_size)
        self.model_config = config.model_config
        self.attn_bs = config.attn_bs
        self.kv_len = config.kv_len

    def compute_cost(self):
        self.qk_flops = (
            2 * self.attn_bs * self.model_config.num_attention_heads * 
            self.kv_len * (2 * self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
        )
        self.softmax_flops =(
            4 * self.attn_bs * self.model_config.num_attention_heads * self.kv_len
        )
        # matrix absorption
        self.uv_absorb_flops =(
            2 * self.attn_bs * self.model_config.kv_lora_rank * 
            self.model_config.num_attention_heads * self.model_config.v_head_dim
        )
        self.compute_time = (
            self.qk_flops / self.cube_flops +
            self.softmax_flops / self.vec_flops +
            self.uv_absorb_flops / self.cube_flops
        )
        return self.compute_time

    def memory_cost(self):
        self.bytes = (
            self.elem_size * self.attn_bs * self.kv_len *
            (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim) +
            self.elem_size * self.model_config.kv_lora_rank *
            self.model_config.num_attention_heads * self.model_config.v_head_dim
        )
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time


class MLAFlashAttentionInt8(BaseOp):
    '''
    Description:
        The Flash Attention operation for the model used MLA attention mechanism in INT8 precision.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config, elem_size=1):
        self.model_config = config.model_config
        self.attn_bs = config.attn_bs
        self.kv_len = config.kv_len
        self.seq_len = config.seq_len
        super().__init__("MLAFlashAttentionInt8", config.aichip_config, elem_size)

    def op_compute_disc(self):
        if self.attn_bs < 32:
            if self.kv_len == 2048:
                return 0.4
            elif self.kv_len == 4096:
                return 0.45
            elif self.kv_len == 8192:
                return 0.5
            else:
                return 0.55
        elif self.attn_bs >= 32 and self.attn_bs < 64:
            if self.kv_len == 2048:
                return 0.45
            elif self.kv_len == 4096:
                return 0.5
            elif self.kv_len == 8192:
                return 0.55
            else:
                return 0.6
        else:
            if self.kv_len == 2048:
                return 0.5
            elif self.kv_len == 4096:
                return 0.55
            elif self.kv_len == 8192:
                return 0.6
            else:
                return 0.65

    def compute_cost(self):
        # qk_position = 2*B*128/TP*S*64*KV
        # FP16, cube core
        # - q_rope = [B, 128/TP, S, 64]
        # - k_rope = [B, 1, KV, 64] (trans) [B, 1, 64, KV]
        # - output_qk = [B, 128/TP, S, KV]
        qk_rope = (
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.model_config.qk_rope_head_dim * self.kv_len
        )
        # qk_matmul = 2*B*128/TP*S*512*KV
        # FP16, cube core
        # - qk = [B, 128/TP, S, 512]
        # - kv_nope = [B, 1, KV, 512] (trans) [B, 1, 512, KV]
        # - output_qkv = [B, 128/TP, S, KV]
        qk_matmul =(
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.model_config.kv_lora_rank * self.kv_len
        )
        # (output_qk + output_qkv) -> safe_softmax = 5*[B, 128/TP, S, KV]
        # FP16, vector core
        # - output_qk = [B, 128/TP, S, KV]
        # - output_qkv = [B, 128/TP, S, KV]
        # - output_softmax = [B, 128/TP, S, KV]
        softmax =(
            5 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.kv_len
        )
        # qkv_matmul = 2*B*128/TP*S*KV*512
        # FP16, cube core
        # - output_softmax = [B, 128/TP, S, KV]
        # - kv_nope = [B, 1, KV, 512]
        # - output_matmul = [B, 128/TP, S, 512]
        sv_matmul =(
            2 * self.attn_bs * self.model_config.num_attention_heads *
            self.seq_len * self.kv_len * self.model_config.kv_lora_rank
        )
        cube_time = qk_rope / self.cube_flops_fp16 + (qk_matmul + sv_matmul)/ self.cube_flops_int8
        vec_time = softmax / self.vec_flops_fp16
        self.compute_time = cube_time + vec_time
        return self.compute_time

    def memory_cost(self):
        self.bytes = (
            self.attn_bs * self.kv_len * 
            (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim) +
            self.model_config.kv_lora_rank*self.model_config.num_attention_heads *
            self.model_config.v_head_dim
        )
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time


class GQAFlashAttentionFP16(BaseOp):
    '''
    Description:
        The Flash Attention operation for the model used GQA attention mechanism in FP16 precision.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config, elem_size=2):
        self.op_compute_disc()
        super().__init__("FlashAttentionFP16", config.aichip_config, elem_size)
        self.model_config = config.model_config
        self.attn_bs = config.attn_bs
        self.kv_len = config.kv_len
        self.seq_len = config.seq_len

    def op_compute_disc(self):
        """
        Dynamic discount factor based on batch size and sequence length.
        
        Hardware utilization varies significantly with input shapes:
        - Small batch: low parallelism, memory latency dominated (~20-30%)
        - Large batch + long context: compute saturated, high utilization (~50-65%)
        
        This function approximates the efficiency curve based on empirical data
        from Ascend NPU attention benchmarks.
        
        Returns:
            float: Discount factor in range [0.20, 0.65]
        """
        # Normalize batch size and kv_len to reference points
        # Reference: bs=64, kv_len=8192 typically achieves ~40% efficiency
        bs_factor = min(1.0, max(0.25, self.attn_bs / 64.0))
        kv_factor = min(1.0, max(0.25, self.kv_len / 8192.0))
        
        # Base efficiency + scaling based on problem size
        # Weight: 60% batch size, 40% sequence length (empirical)
        base_disc = 0.28
        scale = 0.37 * (bs_factor * 0.6 + kv_factor * 0.4)
        
        # Clamp to reasonable range [0.20, 0.65]
        return max(0.20, min(0.65, base_disc + scale))

    def compute_cost(self):
        # qk_matmul: 2*B*n*s*D*kv
        # query_states: [B, n, s, D]
        # key_states: [B, n_kv, kv, D]
        # qk: [B, n, s, kv]
        qk_matmul = (
            2 * self.attn_bs *
            self.model_config.num_heads *
            self.seq_len *
            self.model_config.head_size *
            self.kv_len
        )
        # softmax
        softmax = (
            5 * self.attn_bs *
            self.model_config.num_heads *
            self.seq_len *
            self.kv_len
        )
        # qkv_matmul: 2*B*n*s*kv*D
        # qk: [B, n, s, kv]
        # value_statue: [B, n_kv, kv, D]
        qkv_matmul = (
            2 * self.attn_bs *
            self.model_config.num_heads *
            self.seq_len *
            self.model_config.head_size *
            self.kv_len
        )
        cube_time = (qk_matmul + qkv_matmul) / self.cube_flops_fp16
        vec_time = softmax / self.vec_flops_fp16
        self.compute_time = cube_time + vec_time
        return self.compute_time

    def memory_cost(self):
        # kv_cache
        self.bytes = (
            2 * self.elem_size *
            self.attn_bs *
            self.model_config.kv_heads *
            self.kv_len *
            self.model_config.head_size
        )
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time
