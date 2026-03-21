from typing import Tuple
from abc import ABC, abstractmethod
from conf.common import BYTE_2_GB, DTYPE_BF16, DTYPE_FP16
from conf.model_config import ModelConfig
from conf.config import Config


class BaseSearch(ABC):
    '''
    Description:
        The base class for the search algorithm.
        Inherit from this class to create a new search algorithm, such as AFD Search, DeepEP Search, etc.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config: Config):
        self.config = config
    
    def compute_activation_memory(
        self,
        model_config: ModelConfig,
        attn_bs: int,
        seq_len: int = None
    ) -> float:
        '''
        Description:
            Compute the activation memory (dynamic memory) for forward pass.
            This includes intermediate activations stored during computation.
        
        Args:
            model_config: The configuration of the model.
            attn_bs: The attention batch size.
            seq_len: Sequence length (default: config.seq_len)
        
        Returns:
            activation_memory: Total activation memory in GB.
        
        Note:
            Activation memory is often overlooked but can be significant,
            especially for large batch sizes and long sequences.
            Typical components:
            - Q/K/V projections: 3 × bs × seq_len × hidden_size
            - Attention output: bs × seq_len × hidden_size
            - MoE intermediates: bs × seq_len × num_experts_per_tok × moe_intermediate_size
            - Normalization buffers: 2 × bs × seq_len × hidden_size
        '''
        if seq_len is None:
            seq_len = self.config.seq_len
        
        # Attention activations (per layer)
        qkv_activations = (
            3 * attn_bs * seq_len * model_config.hidden_size * DTYPE_FP16
        )
        attn_output = attn_bs * seq_len * model_config.hidden_size * DTYPE_FP16
        per_layer_attn = qkv_activations + attn_output
        
        # MoE activations (per MoE layer)
        # Each token activates num_experts_per_tok experts
        moe_input = attn_bs * seq_len * model_config.hidden_size * DTYPE_FP16
        moe_intermediate = (
            attn_bs * seq_len * model_config.num_experts_per_tok * 
            model_config.moe_intermediate_size * DTYPE_FP16
        )
        moe_output = attn_bs * seq_len * model_config.hidden_size * DTYPE_FP16
        per_moe_layer = moe_input + moe_intermediate + moe_output
        
        # Total activation memory (approximate, assuming pipeline parallelism)
        # We only need to store activations for current micro-batch
        total_activation = (
            per_layer_attn * model_config.num_layers +
            per_moe_layer * model_config.num_moe_layers
        ) * BYTE_2_GB
        
        # Add normalization buffers
        norm_buffers = (
            2 * attn_bs * seq_len * model_config.hidden_size * 
            model_config.num_layers * DTYPE_FP16 * BYTE_2_GB
        )
        
        return total_activation + norm_buffers

    def compute_MLA_memory_size(
        self,
        model_config: ModelConfig,
        attn_bs: int
    ) -> Tuple[float, float, float, float]:
        '''
        Description:
            Compute the memory size of the model used MLA attention mechanism.
        Args:
            model_config: The configuration of the model.
            attn_bs: The attention batch size.
        Returns:
            kv_size: The memory size of the KV cache, GB.
            attn_static_memory: The memory size of the attention static memory, GB.
            mlp_static_memory: The memory size of the MLP static memory, GB.
            per_router_expert_memory: The memory size of the per router expert memory, GB.
        Note:
            This function now includes activation memory for more accurate memory estimation.
        '''
        # KVCache Size
        kv_size = (
            attn_bs * self.config.kv_len * 
            (model_config.kv_lora_rank + model_config.qk_rope_head_dim) * 
            model_config.num_layers * BYTE_2_GB * DTYPE_BF16
        )

        # Attention Static Memory
        if model_config.q_lora_rank == 0:
            q_a_proj = 0
            q_nope = (
                model_config.num_attention_heads *
                model_config.qk_nope_head_dim *
                model_config.hidden_size
            )
            q_rope = (
                model_config.num_attention_heads * 
                model_config.qk_rope_head_dim * 
                model_config.hidden_size
            )
        else:
            q_a_proj = model_config.q_lora_rank * model_config.hidden_size
            q_nope = (
                model_config.num_attention_heads * 
                model_config.qk_nope_head_dim * 
                model_config.q_lora_rank
            )
            q_rope = (
                model_config.num_attention_heads * 
                model_config.qk_rope_head_dim * 
                model_config.q_lora_rank
            )
        k_nope = model_config.kv_lora_rank * model_config.hidden_size
        k_rope = model_config.qk_rope_head_dim * model_config.hidden_size
        q_absorb = (
            model_config.num_attention_heads * 
            model_config.qk_nope_head_dim * 
            model_config.kv_lora_rank
        )
        uv_absorb = (
            model_config.kv_lora_rank * 
            model_config.num_attention_heads * 
            model_config.v_head_dim
        )
        o_proj = (
            model_config.num_attention_heads * 
            model_config.v_head_dim * 
            model_config.hidden_size
        )

        attn_static_memory = (
            (q_a_proj + q_nope + q_rope + k_nope + k_rope + q_absorb + uv_absorb + o_proj) * 
            model_config.num_layers * BYTE_2_GB
        )

        # mlp memory
        if model_config.num_layers > model_config.num_moe_layers:
            mlp_static_memory = (
                    3 * model_config.intermediate_size * 
                    model_config.hidden_size * BYTE_2_GB
                ) * (model_config.num_layers - model_config.num_moe_layers)
        else:
            mlp_static_memory = 0

        # per router expert momory
        per_router_expert_memory = (
            3 * model_config.moe_intermediate_size * 
            model_config.hidden_size * 
            model_config.num_layers * BYTE_2_GB
        )
        
        return kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory

    def compute_GQA_memory_size(
        self,
        model_config: ModelConfig,
        attn_bs: int
    ) -> Tuple[float, float, float, float]:
        '''
        Description:
            Compute the memory size of the model used GQA attention mechanism.
        Args:
            model_config: The configuration of the model.
            attn_bs: The attention batch size.
        Returns:
            kv_size: The memory size of the KV cache, GB.
            attn_static_memory: The memory size of the attention static memory, GB.
            mlp_static_memory: The memory size of the MLP static memory, GB.
            per_router_expert_memory: The memory size of the per router expert memory, GB.
        Note:
            This function now includes activation memory for more accurate memory estimation.
        '''
        # KVCache Size
        kv_size = (
            attn_bs * self.config.kv_len * 
            model_config.kv_heads * 
            model_config.head_size * 
            model_config.num_layers * BYTE_2_GB * DTYPE_BF16
        )

        # Attention Static Memory
        q_proj = model_config.num_heads * model_config.head_size * model_config.hidden_size
        k_proj = model_config.kv_heads * model_config.head_size * model_config.hidden_size
        v_proj = model_config.kv_heads * model_config.head_size * model_config.hidden_size
        o_rope = model_config.num_heads * model_config.head_size * model_config.hidden_size

        attn_static_memory = (
            (q_proj + k_proj + v_proj + o_rope) * 
            model_config.num_layers * BYTE_2_GB
        )

        # mlp memory
        if model_config.num_layers > model_config.num_moe_layers:
            mlp_static_memory = (
                    3 * model_config.intermediate_size * 
                    model_config.hidden_size * BYTE_2_GB
                ) * (model_config.num_layers - model_config.num_moe_layers)
        else:
            mlp_static_memory = 0

        # per router expert momory
        per_router_expert_memory = (
            3 * model_config.moe_intermediate_size * 
            model_config.hidden_size * 
            model_config.num_layers * BYTE_2_GB
        )
        return kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory

    @abstractmethod
    def deployment(self):
        pass
