from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional
import math

from conf.common import GB_2_BYTE


@dataclass
class NetworkTopologyConfig:
    name: str = "none"
    topology_type: str = "none"
    num_devices: int = 0

    full_mesh_size: int = 0
    full_mesh_link_bw: float = 0.0

    cross_domain_type: str = "none"
    num_domains: int = 0
    cross_domain_num_links: int = 0
    cross_domain_link_bw: float = 0.0
    cross_domain_shared_factor: float = 1.0

    clos_total_bw: float = 0.0

    communication_utilization: float = 0.8
    pcie_cross_domain_utilization: float = 1.0
    comm_kernel_launch_overhead_us: float = 7.0
    moe_unbalance: float = 3.0

    _PRESETS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "none": {
            "name": "none",
            "topology_type": "none",
            "num_devices": 0,
            "full_mesh_size": 0,
            "full_mesh_link_bw": 0.0,
            "cross_domain_type": "none",
            "num_domains": 0,
            "cross_domain_num_links": 0,
            "cross_domain_link_bw": 0.0,
            "cross_domain_shared_factor": 1.0,
            "clos_total_bw": 0.0,
            "communication_utilization": 0.8,
            "pcie_cross_domain_utilization": 1.0,
            "comm_kernel_launch_overhead_us": 7.0,
            "moe_unbalance": 3.0,
        },
        "mgx_16": {
            "name": "mgx_16",
            "topology_type": "mgx",
            "num_devices": 16,
            "full_mesh_size": 4,
            "full_mesh_link_bw": 50.0 * GB_2_BYTE,
            "cross_domain_type": "clos",
            "num_domains": 4,
            "cross_domain_num_links": 4,
            "cross_domain_link_bw": 50.0 * GB_2_BYTE,
            "cross_domain_shared_factor": 1.0,
            "clos_total_bw": 0.0,
            "communication_utilization": 0.8,
            "pcie_cross_domain_utilization": 1.0,
            "comm_kernel_launch_overhead_us": 7.0,
            "moe_unbalance": 3.0,
        },
        "pcie_16": {
            "name": "pcie_16",
            "topology_type": "pcie",
            "num_devices": 16,
            "full_mesh_size": 4,
            "full_mesh_link_bw": 50.0 * GB_2_BYTE,
            "cross_domain_type": "pcie",
            "num_domains": 4,
            "cross_domain_num_links": 3,
            "cross_domain_link_bw": 64.0 * GB_2_BYTE,
            "cross_domain_shared_factor": 4.0,
            "clos_total_bw": 0.0,
            "communication_utilization": 0.8,
            "pcie_cross_domain_utilization": 0.8,
            "comm_kernel_launch_overhead_us": 7.0,
            "moe_unbalance": 3.0,
        },
        "clos_16_400gbps": {
            "name": "clos_16_400gbps",
            "topology_type": "clos",
            "num_devices": 16,
            "full_mesh_size": 16,
            "full_mesh_link_bw": 0.0,
            "cross_domain_type": "none",
            "num_domains": 1,
            "cross_domain_num_links": 0,
            "cross_domain_link_bw": 0.0,
            "cross_domain_shared_factor": 1.0,
            "clos_total_bw": 400.0 * GB_2_BYTE,
            "communication_utilization": 0.8,
            "pcie_cross_domain_utilization": 1.0,
            "comm_kernel_launch_overhead_us": 7.0,
            "moe_unbalance": 3.0,
        },
    }

    @classmethod
    def from_preset(cls, name: str) -> "NetworkTopologyConfig":
        if name not in cls._PRESETS:
            raise ValueError(f"Unknown topology preset '{name}'. Available: {', '.join(cls._PRESETS.keys())}")
        return cls(**cls._PRESETS[name])

    @classmethod
    def list_presets(cls) -> List[str]:
        return list(cls._PRESETS.keys())

    def get_intra_domain_bandwidth(self) -> float:
        if self.topology_type == "clos" and self.clos_total_bw > 0:
            return self.clos_total_bw
        if self.full_mesh_size <= 1:
            return 0.0
        return (self.full_mesh_size - 1) * self.full_mesh_link_bw

    def get_cross_domain_bandwidth(self) -> float:
        if self.topology_type == "clos" and self.clos_total_bw > 0:
            return 0.0
        if self.num_domains <= 1 or self.cross_domain_num_links <= 0:
            return 0.0
        return (self.cross_domain_num_links * self.cross_domain_link_bw) / max(1.0, self.cross_domain_shared_factor)

    def get_effective_bandwidth(self, *, comm_type: str = "all2all", group_size: Optional[int] = None) -> float:
        if self.topology_type == "clos" and self.clos_total_bw > 0:
            return self.get_intra_domain_bandwidth() * self.communication_utilization

        if group_size is None or group_size <= 0:
            group_size = self.num_devices

        intra_bw = self.get_intra_domain_bandwidth()
        cross_bw = self.get_cross_domain_bandwidth()

        if self.full_mesh_size > 0 and group_size <= self.full_mesh_size:
            return intra_bw * self.communication_utilization

        if comm_type == "all2all":
            effective_bw = min(intra_bw, cross_bw) if cross_bw > 0 else intra_bw
        else:
            effective_bw = cross_bw if cross_bw > 0 else intra_bw

        util = self.communication_utilization
        if self.topology_type == "pcie" and cross_bw > 0:
            util *= self.pcie_cross_domain_utilization

        return effective_bw * util

    def get_all2all_effective_bytes(self, bytes_per_device: float, *, group_size: int, moe_unbalance: Optional[float] = None) -> float:
        if group_size <= 1:
            return 0.0
        unbalance = moe_unbalance if moe_unbalance is not None else self.moe_unbalance
        unbalance_factor = math.sqrt(unbalance) if unbalance and unbalance > 1.0 else 1.0
        return bytes_per_device * (group_size - 1) / group_size * unbalance_factor

