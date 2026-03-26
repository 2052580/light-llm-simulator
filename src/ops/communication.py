from src.ops.base import BaseOp
from conf.common import US_2_SEC

class Dispatch(BaseOp):
    '''
    Description:
        The dispatch operation.
        It is used to route the tokens to the experts.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config, elem_size=1):
        self.config = config
        super().__init__("Dispatch", config.aichip_config, elem_size)
        self.model_config = config.model_config

    def op_memory_disc(self):
        return 0.7

    def memory_cost(self):
        if self.config.serving_mode == "DeepEP":
            dispatch_packet = (
                self.config.attn_bs *
                self.config.seq_len *
                self.model_config.hidden_size *
                self.model_config.num_experts_per_tok *
                self.elem_size
            )
        elif self.config.serving_mode == "AFD":
            dispatch_packet = (
                self.config.attn_bs *
                self.config.seq_len *
                self.model_config.hidden_size *
                self.model_config.num_experts_per_tok *
                self.elem_size *
                self.config.attn_die /
                self.config.ffn_die
            )
        group_size = int(self.config.ffn_die)
        if hasattr(self.config, "topology_config") and self.config.topology_config is not None and self.config.topology_config.topology_type != "none":
            bytes_per_device = float(dispatch_packet)
            effective_bytes = self.config.topology_config.get_all2all_effective_bytes(
                bytes_per_device,
                group_size=group_size,
            )
            bw = float(self.config.topology_config.get_effective_bandwidth(comm_type="all2all", group_size=group_size))
            bw *= float(self.op_memory_disc())
            if bw > 0:
                self.memory_time = effective_bytes / bw + float(self.config.topology_config.comm_kernel_launch_overhead_us) * US_2_SEC
            else:
                self.memory_time = float("inf")
        else:
            self.memory_time = dispatch_packet / self.inter_node_bandwidth
        return self.memory_time

class Combine(BaseOp):
    '''
    Description:
        The combine operation.
        It is used to aggregate the outputs of the experts.
    Attributes:
        config: The configuration of the search task.
    '''
    def __init__(self, config, elem_size=2):
        self.config = config
        super().__init__("Dispatch", config.aichip_config, elem_size)
        self.model_config = config.model_config

    def op_memory_disc(self):
        return 0.7

    def memory_cost(self):
        if self.config.serving_mode == "DeepEP":
            combine_packet = (
                self.config.attn_bs *
                self.config.seq_len *
                self.model_config.hidden_size *
                self.model_config.num_experts_per_tok *
                self.elem_size
            )
        elif self.config.serving_mode == "AFD":
            combine_packet = (
                self.config.attn_bs *
                self.config.seq_len *
                self.model_config.hidden_size *
                self.model_config.num_experts_per_tok *
                self.elem_size *
                self.config.attn_die /
                self.config.ffn_die
            )
        group_size = int(self.config.ffn_die)
        if hasattr(self.config, "topology_config") and self.config.topology_config is not None and self.config.topology_config.topology_type != "none":
            bytes_per_device = float(combine_packet)
            effective_bytes = self.config.topology_config.get_all2all_effective_bytes(
                bytes_per_device,
                group_size=group_size,
            )
            bw = float(self.config.topology_config.get_effective_bandwidth(comm_type="all2all", group_size=group_size))
            bw *= float(self.op_memory_disc())
            if bw > 0:
                self.memory_time = effective_bytes / bw + float(self.config.topology_config.comm_kernel_launch_overhead_us) * US_2_SEC
            else:
                self.memory_time = float("inf")
        else:
            self.memory_time = combine_packet / self.inter_node_bandwidth
        return self.memory_time
