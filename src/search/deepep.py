import logging
import math
import os
import pandas as pd
from conf.common import MIN_ROUTED_EXPERT_PER_DIE, US_2_MS, SEC_2_US, BYTE_2_GB, MEMORY_THRESHOLD_RATIO, MS_2_SEC, MS_2_US
from src.search.base import BaseSearch
from src.model.register import get_model, get_attention_family
from conf.config import Config


class DeepEpSearch(BaseSearch):
    '''
    Description:
        The DeepEP search algorithm.
        It is used to search the optimal attention batch size for the model used DeepEP serving.
    Attributes:
        config: The configuration of the DeepEP search task.
        perf_deepep_results: The performance results of the DeepEP search,
        it contains the following columns:
            attn_bs: Attention batch size, int.
            ffn_bs: FFN batch size(tokens per die), float.
            kv_len: KV cache length, int.
            total_die: The number of Total die, int.
            attn_time: Attention time for per layer (μs), float.
            mlp_time: MLP time for per dense layer (μs), float.
            moe_time: MoE time for per layer (μs), float.
            commu_time: Communication time for per layer (μs), float.
            dispatch_time: Dispatch time for per layer (μs), float.
            combine_time: Combine time for per layer (μs), float.
            e2e_time: End-to-end time (ms), float.
            throughput: Throughput (tokens/second), float.
    '''
    def __init__(self, config: Config):
        super().__init__(config)
        self.perf_deepep_results = []

    def search_bs(self):
        '''
        Description:
            Search the optimal attention batch size for the model used DeepEP serving.
        '''
        min_total_die, max_total_die, die_step = self.config.min_die, self.config.max_die, self.config.die_step
        for total_die in range(min_total_die, max_total_die, die_step):
            routed_expert_per_die = self.config.model_config.n_shared_experts + max(
                MIN_ROUTED_EXPERT_PER_DIE,
                math.ceil(self.config.model_config.n_routed_experts / total_die)
            )
            attn_bs_min, attn_bs_max = self.config.min_attn_bs, self.config.max_attn_bs

            # search max attention bs
            while attn_bs_max - attn_bs_min > 1:
                attn_bs = (attn_bs_min + attn_bs_max) // 2
                if get_attention_family(self.config.model_type) == "MLA":
                    kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_MLA_memory_size(
                        self.config.model_config, attn_bs
                    )
                elif get_attention_family(self.config.model_type) == "GQA":
                    kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_GQA_memory_size(
                        self.config.model_config, attn_bs
                    )
                latency_constraint = (
                    self.config.tpot * MS_2_US / self.config.micro_batch_num * (1 + self.config.multi_token_ratio)
                )
                attn_die, ffn_die = total_die, total_die,
                ffn_bs = attn_bs * self.config.model_config.num_experts_per_tok
                self.config.attn_bs = attn_bs
                self.config.ffn_bs = ffn_bs
                self.config.attn_die = attn_die
                self.config.ffn_die = ffn_die
                self.config.routed_expert_per_die = routed_expert_per_die
                model = get_model(self.config)
                attn = model["attn"]
                attn()
                moe = model["moe"]
                moe()
                attn_time = attn.e2e_time * SEC_2_US
                moe_time = moe.e2e_time * SEC_2_US
                commu_time = moe.commu_time * SEC_2_US
                dispatch_time = moe.dispatch_time * SEC_2_US
                combine_time = moe.combine_time * SEC_2_US

                ffn_dynamic_memory = (
                    ffn_bs * self.config.model_config.hidden_size * 
                    self.config.model_config.num_layers * BYTE_2_GB
                )
                ffn_static_memory = per_router_expert_memory * routed_expert_per_die
                total_memory = (
                    kv_size * self.config.micro_batch_num + attn_static_memory + 
                    mlp_static_memory + ffn_dynamic_memory + ffn_static_memory
                )
                e2e_time_per_moe_layer = attn_time + moe_time + commu_time
                e2e_time = e2e_time_per_moe_layer * self.config.model_config.num_moe_layers

                if self.config.model_config.num_layers > self.config.model_config.num_moe_layers:
                    # compute per dense layer time
                    mlp = model["mlp"]
                    mlp()
                    mlp_time = mlp.e2e_time * SEC_2_US
                    e2e_time_per_dense_layer = attn_time + mlp_time
                    e2e_time = e2e_time + e2e_time_per_dense_layer * self.config.model_config.first_k_dense_replace
                else:
                    mlp_time = 0.0
                    e2e_time_per_dense_layer = 0.0
                if (e2e_time > latency_constraint or 
                    total_memory > self.config.aichip_config.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO):
                    attn_bs_max = attn_bs
                else:
                    attn_bs_min = attn_bs

            e2e_time = e2e_time * US_2_MS
            throughput = attn_bs / e2e_time / MS_2_SEC * (1 + self.config.multi_token_ratio)

            logging.info(f"-------DeepEP Search Result:-------")
            logging.info(
                f"attn_bs:{attn_bs}, ffn_bs:{ffn_bs}, "
                f"kv_len:{self.config.kv_len}, total_die:{total_die}, "
                f"attn_time:{attn_time} us, moe_time:{moe_time} us, "
                f"commu_time:{commu_time} us, dispatch_time:{dispatch_time} us, combine_time:{combine_time} us, "
                f"e2e_time:{e2e_time} ms, throughput:{throughput} tokens/die/s, "
                f"e2e_time_per_dense_layer:{e2e_time_per_dense_layer} us, e2e_time_per_moe_layer:{e2e_time_per_moe_layer} us, "
                f"kv_size:{kv_size} GB, attn_static_memory:{attn_static_memory} GB, "
                f"mlp_static_memory:{mlp_static_memory} GB, ffn_static_memory:{ffn_static_memory} GB"
            )

            self.perf_deepep_results.append([
                attn_bs, ffn_bs, self.config.kv_len, total_die,
                attn_time, moe_time, commu_time, dispatch_time, combine_time, e2e_time,
                e2e_time_per_dense_layer, e2e_time_per_moe_layer, throughput,
                kv_size, attn_static_memory, mlp_static_memory, ffn_static_memory
            ])

        columns = [
            'attn_bs', 'ffn_bs', 'kv_len', 'total_die', 'attn_time(us)', 
            'moe_time(us)', 'commu_time(us)', 'dispatch_time(us)', 'combine_time(us)', 'e2e_time(ms)',
            'e2e_time_per_dense_layer(us)', 'e2e_time_per_moe_layer(us)', 'throughput(tokens/die/s)',
            'kv_size(GB)', 'attn_static_memory(GB)', 'mlp_static_memory(GB)', 'ffn_static_memory(GB)'
        ]
        df = pd.DataFrame(self.perf_deepep_results, columns=columns)
        result_dir = f"data/deepep/"
        file_name = f"{self.config.device_type.name}-{self.config.model_type.name}-tpot{int(self.config.tpot)}-kv_len{self.config.kv_len}.csv"
        os.makedirs(result_dir, exist_ok=True)
        result_path = result_dir + file_name
        df.to_csv(result_path, index=False)

    def deployment(self):
        self.search_bs()
