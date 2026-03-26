import logging
from conf.common import MAX_AVG_RATIO
from conf.config import Config
from src.model.base import BaseModule
from src.ops import (
    OpGeMatmul,
    OpQuantBatchMatmul,
    OpRotary,
    GQAFlashAttentionFP16,
    Dispatch,
    Combine,
    OpGroupedMatmul,
    OpSwiglu,
    OpNorm,
    OpGDNLinear,
)


class Qwen35DecodeAttn(BaseModule):
    def __init__(self, config: Config):
        super().__init__(config)
        self.attn_bs = config.attn_bs
        self._build_ops()

    def _build_ops(self):
        bs = self.attn_bs * self.config.seq_len
        mconf = self.model_config
        self.query_states = OpGeMatmul(
            "query_states",
            bs,
            mconf.hidden_size,
            mconf.num_heads * mconf.head_size,
            self.aichip_config
        )
        self.key_states = OpGeMatmul(
            "key_states",
            bs,
            mconf.hidden_size,
            mconf.kv_heads * mconf.head_size,
            self.aichip_config
        )
        self.value_states = OpGeMatmul(
            "value_states",
            bs,
            mconf.hidden_size,
            mconf.kv_heads * mconf.head_size,
            self.aichip_config
        )
        self.query_rope = OpRotary(
            "query_rope",
            bs,
            mconf.num_heads,
            self.config.seq_len,
            mconf.head_size,
            self.aichip_config
        )
        self.key_rope = OpRotary(
            "key_rope",
            bs,
            mconf.kv_heads,
            self.config.seq_len,
            mconf.head_size,
            self.aichip_config
        )
        self.full_attn = GQAFlashAttentionFP16(self.config)
        self.linear_attn = OpGDNLinear(self.config)
        self.bmm_o_proj = OpQuantBatchMatmul(
            "bmm_o_proj",
            bs,
            mconf.num_heads * mconf.head_size,
            mconf.hidden_size,
            self.aichip_config
        )
        self.norm = OpNorm(self.attn_bs, self.aichip_config)
        self.ops = [
            self.query_states,
            self.key_states,
            self.value_states,
            self.query_rope,
            self.key_rope,
            self.full_attn,
            self.linear_attn,
            self.bmm_o_proj,
            self.norm
        ]

    def _aggregate_times(self):
        r_full = 1.0 / max(1, getattr(self.model_config, "full_attention_interval", 1))
        r_lin = 1.0 - r_full
        attn_e2e = r_full * self.full_attn.e2e_time + r_lin * self.linear_attn.e2e_time
        attn_comp = r_full * self.full_attn.compute_time + r_lin * self.linear_attn.compute_time
        attn_mem = r_full * self.full_attn.memory_time + r_lin * self.linear_attn.memory_time
        self.e2e_time = (
            self.query_states.e2e_time +
            self.key_states.e2e_time +
            self.value_states.e2e_time +
            self.query_rope.e2e_time +
            self.key_rope.e2e_time +
            attn_e2e +
            self.bmm_o_proj.e2e_time +
            self.norm.e2e_time
        )
        self.compute_time = (
            self.query_states.compute_time +
            self.key_states.compute_time +
            self.value_states.compute_time +
            self.query_rope.compute_time +
            self.key_rope.compute_time +
            attn_comp +
            self.bmm_o_proj.compute_time +
            self.norm.compute_time
        )
        self.memory_time = (
            self.query_states.memory_time +
            self.key_states.memory_time +
            self.value_states.memory_time +
            self.query_rope.memory_time +
            self.key_rope.memory_time +
            attn_mem +
            self.bmm_o_proj.memory_time +
            self.norm.memory_time
        )

        logging.info(
            f"Attention Module - attn_bs: {self.config.attn_bs}, "
            f"query_states: {self.query_states.e2e_time * 1e6:.2f}us, "
            f"key_states: {self.key_states.e2e_time * 1e6:.2f}us, "
            f"value_states: {self.value_states.e2e_time * 1e6:.2f}us, "
            f"query_rope: {self.query_rope.e2e_time * 1e6:.2f}us, "
            f"key_rope: {self.key_rope.e2e_time * 1e6:.2f}us, "
            f"full_attn: {self.full_attn.e2e_time * 1e6:.2f}us, "
            f"linear_attn: {self.linear_attn.e2e_time * 1e6:.2f}us, "
            f"bmm_o_proj: {self.bmm_o_proj.e2e_time * 1e6:.2f}us, "
            f"norm: {self.norm.e2e_time * 1e6: .2f}us"
        )


class Qwen35DecodeMoe(BaseModule):
    def __init__(self, config: Config):
        super().__init__(config)
        self.tokens_per_ffn_die = config.ffn_bs * config.seq_len
        self.routed_expert_per_die = config.routed_expert_per_die
        self.commu_time = 0.0
        self.dispatch_time = 0.0
        self.combine_time = 0.0
        self._build_ops()

    def _build_ops(self):
        mconf = self.model_config
        self.dispatch = Dispatch(self.config)
        self.moe_up = OpGroupedMatmul(
            "Qwen35MoEUP",
            self.routed_expert_per_die,
            self.tokens_per_ffn_die,
            mconf.hidden_size,
            2 * mconf.moe_intermediate_size / self.config.ffn_tensor_parallel,
            self.aichip_config,
            elem_size=1
        )
        self.moe_swiglu = OpSwiglu(
            self.tokens_per_ffn_die,
            2 * mconf.moe_intermediate_size,
            self.aichip_config
        )
        self.moe_down = OpGroupedMatmul(
            "Qwen35MoEDown",
            self.routed_expert_per_die,
            self.tokens_per_ffn_die,
            mconf.moe_intermediate_size / self.config.ffn_tensor_parallel,
            mconf.hidden_size,
            self.aichip_config,
            elem_size=1
        )
        self.combine = Combine(self.config)
        self.ops = [self.dispatch, self.moe_up, self.moe_swiglu, self.moe_down, self.combine]

    def _aggregate_times(self):
        self.dispatch_time = self.dispatch.e2e_time
        self.e2e_time = (
            self.moe_up.e2e_time * MAX_AVG_RATIO +
            self.moe_swiglu.e2e_time +
            self.moe_down.e2e_time * MAX_AVG_RATIO
        )
        self.compute_time = (
            self.moe_up.compute_time * MAX_AVG_RATIO +
            self.moe_swiglu.compute_time +
            self.moe_down.compute_time * MAX_AVG_RATIO
        )
        self.memory_time = (
            self.moe_up.memory_time * MAX_AVG_RATIO +
            self.moe_swiglu.memory_time +
            self.moe_down.memory_time * MAX_AVG_RATIO
        )
        self.combine_time = self.combine.e2e_time

        self.commu_time = self.dispatch_time + self.combine_time

        logging.info(
            f"MoE Module - ffn_bs: {self.config.ffn_bs}, "
            f"moe_up: {self.moe_up.e2e_time * 1e6:.2f}us, "
            f"moe_swiglu: {self.moe_swiglu.e2e_time * 1e6:.2f}us, "
            f"moe_down: {self.moe_down.e2e_time * 1e6:.2f}us, "
            f"dispatch_time: {self.dispatch_time * 1e6:.2f}us, "
            f"combine_time: {self.combine_time * 1e6:.2f}us, "
            f"commu_time: {self.commu_time * 1e6:.2f}us"
        )
