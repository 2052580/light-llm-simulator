from conf.model_config import ModelConfig, ModelType
from conf.hardware_config import HWConf, DeviceType
from conf.common import MIN_ROUTED_EXPERT_PER_DIE
from conf.network_topology_config import NetworkTopologyConfig
import math
from typing import Optional


class Config:
    def __init__(
        self,
        serving_mode: str,
        model_type: ModelType,
        device_type: DeviceType,
        min_attn_bs: int,
        max_attn_bs: int,
        min_die: int,
        max_die: int,
        die_step: int,
        tpot:list[int],
        kv_len: list[int],
        micro_batch_num: list[int],
        next_n: int,
        multi_token_ratio: float,
        attn_tensor_parallel: int,
        ffn_tensor_parallel: int,
        topology: Optional[str] = None,
    ) -> None:
        """
        Initialize a Config object.
        A Config object contains all configurations of the search task.
        TODO:
        Allow passing in a yaml file to do patch

        Args:
            serving_mode: The serving mode of the task.
            model_type: The type of the model.
            device_type: The type of the device.
            min_attn_bs: The min number of attention batch size to explore.
            max_attn_bs: The max number of attention batch size to explore.
            min_die: The min number of die to explore.
            max_die: The max number of die to explore.
            die_step: The step size of the die to explore.
            tpot: The target TPOT.
            kv_len: The input sequence length.
            micro_batch_num: The micro batch number.
            next_n: Predict the next n tokens through the MTP(Multi-Token Prediction) technique.
            multi_token_ratio: The acceptance rate of the additionally predicted token.
            attn_tensor_parallel: Number of dies used for tensor model parallelism.
            ffn_tensor_parallel: Number of dies used for tensor model parallelism.
        """
        self.serving_mode = serving_mode
        model_type = ModelType(model_type)
        self.model_type = model_type
        self.model_config = ModelConfig.create_model_config(model_type)
        self.device_type = DeviceType(device_type)
        self.aichip_config = HWConf.create(self.device_type)
        self.topology = topology or "none"
        self.topology_config = NetworkTopologyConfig.from_preset(self.topology)
        self.min_attn_bs = min_attn_bs
        self.max_attn_bs = max_attn_bs
        self.min_die = min_die
        self.max_die = max_die
        self.die_step = die_step
        self.tpot = tpot
        self.kv_len = kv_len
        self.micro_batch_num = micro_batch_num
        self.seq_len = next_n + 1
        self.multi_token_ratio = multi_token_ratio
        self.attn_tensor_parallel = attn_tensor_parallel
        self.ffn_tensor_parallel = ffn_tensor_parallel
        self.attn_bs = min_attn_bs
        self.ffn_bs = self.attn_bs * self.model_config.num_experts_per_tok
        self.attn_die = min_die
        self.ffn_die = min_die
        self.routed_expert_per_die = max(
                MIN_ROUTED_EXPERT_PER_DIE,
                math.ceil(self.model_config.n_routed_experts / self.ffn_die)
            )
