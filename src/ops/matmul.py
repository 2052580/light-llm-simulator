from src.ops.base import BaseOp


class OpGeMatmul(BaseOp):
    '''
    Description:
        The general matrix multiplication operation.
        It is used to multiply two matrices of shape (m, n) and (n, k).
    Attributes:
        name: The name of the operation.
        m: The number of rows of the first matrix.
        n: The number of columns of the first matrix.
        k: The number of columns of the second matrix.
        aichip_config: The hardware configuration.
    '''
    def __init__(self, name, m, n, k, aichip_config, elem_size=2):
        self.m = m
        self.n = n
        self.k = k
        super().__init__(name, aichip_config, elem_size)

    def op_compute_disc(self):
        if self.m < 64:
            return 0.18
        elif self.m >= 64 and self.m < 160:
            return 0.24
        elif self.m >= 160 and self.m < 224:
            return 0.27
        elif self.m >= 224 and self.m < 320:
            return 0.37
        return 0.43

    def compute_cost(self):
        self.compute_flops = 2 * self.m * self.n * self.k
        self.compute_time = self.compute_flops / self.cube_flops
        return self.compute_time

    def memory_cost(self):
        self.bytes = self.elem_size * self.n * self.k + self.elem_size * self.m * self.n
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time


class OpQuantBatchMatmul(BaseOp):
    '''
    Description:
        The quantized batch matrix multiplication operation.
        It is used to multiply two matrices of shape (m, n) and (n, k).
    Attributes:
        name: The name of the operation.
        m: The number of rows of the first matrix.
        n: The number of columns of the first matrix.
        k: The number of columns of the second matrix.
        aichip_config: The hardware configuration.
    '''
    def __init__(self, name, m, n, k, aichip_config, elem_size=1):
        self.m = m
        self.n = n
        self.k = k
        super().__init__(name, aichip_config, elem_size)

    def op_compute_disc(self):
        if self.m < 128:
            return 0.35
        elif self.m >= 128 and self.m < 160:
            return 0.45
        elif self.m >= 160:
            return 0.5

    def compute_cost(self):
        self.compute_flops = 2 * self.m * self.n * self.k
        self.compute_time = self.compute_flops / self.cube_flops_int8
        return self.compute_time

    def memory_cost(self):
        self.bytes = self.elem_size * self.n * self.k + self.elem_size * self.m * self.n
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time


class OpGroupedMatmul(BaseOp):
    '''
    Description:
        The grouped matrix multiplication operation.
        It is used to multiply two matrices of shape (m, n) and (n, k).
    Attributes:
        name: The name of the operation.
        num_experts: The number of experts.
        bs: The FFN batch size.
        m: The number of columns of the first matrix.
        n: The number of columns of the second matrix.
        aichip_config: The hardware configuration.
        elem_size: The element size of the data.
    '''
    def __init__(self, name, num_experts, bs, m, n, aichip_config, elem_size=1):
        self.num_experts = num_experts
        self.bs = bs
        self.m = m
        self.n = n
        super().__init__(name, aichip_config, elem_size)

    def op_memory_disc(self):
        if self.num_experts == 2 or self.num_experts == 3:
            if self.bs <= 256:
                return 0.42
            elif self.bs > 256 and self.bs <= 512:
                return 0.5
            elif self.bs > 512 and self.bs <= 768:
                return 0.4
            elif self.bs > 768 and self.bs <= 1024:
                return 0.3
            elif self.bs > 1024 and self.bs <= 1280:
                return 0.26
            elif self.bs > 1280 and self.bs <= 1536:
                return 0.23
            elif self.bs > 1536 and self.bs <= 1792:
                return 0.195
            elif self.bs > 1792 and self.bs <= 2048:
                return 0.18
            else:
                return 0.16
        if self.num_experts == 4 or self.num_experts == 5:
            if self.bs <= 512:
                return 0.55
            elif self.bs > 512 and self.bs <= 768:
                return 0.5
            elif self.bs > 768 and self.bs <= 1024:
                return 0.37
            elif self.bs > 1024 and self.bs <= 1536:
                return 0.275
            elif self.bs > 1536 and self.bs <= 2048:
                return 0.25
            else:
                return 0.21
        if self.num_experts == 6 and self.num_experts == 7:
            if self.bs <= 768:
                return 0.58
            elif self.bs > 768 and self.bs <= 1536:
                return 0.34
            elif self.bs > 1536 and self.bs <= 3072:
                return 0.26
            else:
                return 0.20
        if self.num_experts == 8 and self.num_experts == 9:
            if self.bs <= 1024:
                return 0.58
            elif self.bs > 1024 and self.bs <= 2048:
                return 0.36
            elif self.bs > 2048 and self.bs <= 4096:
                return 0.26
            else:
                return 0.20
        if self.num_experts > 9 and self.num_experts <= 12:
            if self.bs <= 1536:
                return 0.6
            else:
                return 0.35
        if self.num_experts > 12:
            if self.bs <= 2048:
                return 0.6
            elif self.bs > 2048 and self.bs <= 4096:
                return 0.35
            else:
                return 0.25
        return 0.55

    def compute_cost(self):
        self.compute_flops = 2 * self.bs * self.m * self.n
        self.compute_time = self.compute_flops / self.cube_flops
        return self.compute_time

    def memory_cost(self):
        # Fix: In MoE expert parallelism, weights are distributed across FFN dies.
        # Each die only loads its local expert weights, not all experts.
        # The num_experts here represents total experts, but we only load 1/die_count of them.
        # Note: This assumes expert parallelism where each die handles num_experts/ffn_die experts.
        # For single-die or when expert distribution is handled externally, this gives upper bound.
        self.bytes = (
            self.elem_size * self.bs * self.m +  # Input activation
            self.elem_size * self.m * self.n  # Local expert weights (not all experts)
        )
        # Note: Communication overhead for expert dispatch/combine is calculated separately
        # in Dispatch/Combine operators using inter_node_bandwidth
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time
