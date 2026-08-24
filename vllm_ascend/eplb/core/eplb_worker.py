#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from multiprocessing import Process, Queue
from typing import Any

import numpy as np
import torch
from vllm.distributed import get_ep_group
from vllm.logger import logger

from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map
from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory


class EplbWorker:
    def __init__(
        self,
        shared_dict,
        policy_type,
        enable_d2d: bool = True,
        tp_size: int | None = None,
    ):
        self.policy_type = policy_type
        self.policy = PolicyFactory.generate_policy(policy_type)
        self.shared_dict = shared_dict
        self.old_expert_maps = None
        self.enable_d2d = enable_d2d
        self.tp_size = tp_size
        self.rank_id = get_ep_group().rank_in_group
        self.multi_stage = policy_type == 3
        # ms-service-metric: latest cycle snapshot for Ascend dynamic EPLB handlers.
        self.latest_expert_hotness = None
        self.latest_rebalance_result = None
        self.latest_load_balance = None
        self.latest_map_consistency = None

    def do_update(self):
        # put data in to queue
        # in process self.policy.generate_policy()
        # get epxert table && tensor

        # async stream
        # D2D
        # H2D
        # Get initial expert_map
        torch.set_num_threads(1)
        if self.old_expert_maps is None:
            self.old_expert_maps = self.get_init_expert_maps()
            if self.old_expert_maps is not None:
                self.num_local_experts = self.old_expert_maps.max() + 1
            else:
                raise ValueError("Failed to get expert_maps from shared_dict.")

        # Get MOE load information
        load_info = self.fetch_and_sum_load_info()
        if load_info is None:
            logger.debug("[eplb/worker] No moe_load data available yet, skipping this cycle")
            # ms-service-metric begin
            self.latest_rebalance_result = {
                "result": "skipped",
                "policy_type": int(self.policy_type),
                "fallback_layers": 0,
            }
            # ms-service-metric end
            return

        # Get the updated expert table based on the workload information
        old_placement = self.global2local(self.old_expert_maps, self.num_local_experts)
        _, _, new_placement = self.calculate_rebalance_experts(load_info, old_placement)

        if self.rank_id == 0:
            if self.multi_stage:
                hotness = self._calculate_hotness(old_placement, load_info.sum(0))
            else:
                hotness = self._calculate_hotness(old_placement, load_info)
            # ms-service-metric begin: expose EPLB hotness details for metrics collection.
            current_mean, current_max, current_imbalance_list = self._compute_imbalance(
                old_placement, hotness, return_list=True
            )
            update_mean, update_max, update_imbalance_list = self._compute_imbalance(
                new_placement, hotness, return_list=True
            )
            self.latest_expert_hotness = {
                "current_mean": current_mean,
                "current_max": current_max,
                "update_mean": update_mean,
                "update_max": update_max,
                "current_imbalance_list": current_imbalance_list,
                "update_imbalance_list": update_imbalance_list,
            }
            self.latest_load_balance = self._compute_load_balance(load_info)
            # ms-service-metric end.
            logger.info(
                "[eplb/worker] Expert hotness imbalance, current: mean=%.3f max=%.3f, updated: mean=%.3f max=%.3f",
                current_mean,
                current_max,
                update_mean,
                update_max,
            )

        if not torch.is_tensor(new_placement):
            new_placement = torch.tensor(new_placement)
        placement_stats = self.check_expert_placement(old_placement, new_placement)
        # ms-service-metric begin
        fallback_layers = int(placement_stats.get("fallback_layers", 0))
        self.latest_map_consistency = {
            "fallback_layers": fallback_layers,
            "duplicate_count": int(placement_stats.get("duplicate_count", 0)),
            "missing_count": int(placement_stats.get("missing_count", 0)),
        }
        self.latest_rebalance_result = {
            "result": "fallback" if fallback_layers > 0 else "success",
            "policy_type": int(self.policy_type),
            "fallback_layers": fallback_layers,
        }
        # ms-service-metric end
        new_expert_maps = self.local2global(new_placement)
        self.update_expert_map(new_expert_maps)

        update_info = self.compose_expert_update_info_greedy(new_expert_maps, self.old_expert_maps)
        self.old_expert_maps = new_expert_maps
        logger.debug("[eplb/worker] EPLB Process compute complete")

        packed_update_info = self.pack_update_info(update_info)

        return packed_update_info

    def check_expert_placement(self, old_placement, new_placement):
        num_layers = old_placement.shape[0]
        num_ranks = old_placement.shape[1]
        fallback_layers = 0
        duplicate_count = 0
        missing_count = 0

        for layer_id in range(num_layers):
            # check if any logical expert is not placed on any rank
            old_unique = int(torch.unique(old_placement[layer_id]).numel())
            new_unique = int(torch.unique(new_placement[layer_id]).numel())
            if new_unique < old_unique:
                missing_count += old_unique - new_unique
                logger.error("[eplb/worker] There exists expert not placed on any rank in layer %s", layer_id)
                new_placement[layer_id] = old_placement[layer_id]
                fallback_layers += 1
                continue

            layer_fell_back = False
            for rank_id in range(num_ranks):
                new_placement_check = new_placement[layer_id][rank_id]
                old_placement_check = old_placement[layer_id][rank_id]

                # check if same logical experts are placed on the same NPU
                if new_placement_check.numel() != torch.unique(new_placement_check).numel():
                    duplicate_count += int(new_placement_check.numel() - torch.unique(new_placement_check).numel())
                    logger.error(
                        "[eplb/worker] Replicated experts are placed on the same NPU; "
                        "expert placement on layer %s, rank %s is invalid",
                        layer_id,
                        rank_id,
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    layer_fell_back = True
                    break

                # check if there is any experts movement inside one NPU
                expert_not_move = torch.isin(new_placement_check, old_placement_check)
                if not torch.equal(new_placement_check[expert_not_move], old_placement_check[expert_not_move]):
                    logger.error(
                        "[eplb/worker] Expert movement inside NPU detected; "
                        "expert placement on layer %s, rank %s is invalid",
                        layer_id,
                        rank_id,
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    layer_fell_back = True
                    break
            if layer_fell_back:
                fallback_layers += 1

        return {
            "fallback_layers": fallback_layers,
            "duplicate_count": duplicate_count,
            "missing_count": missing_count,
        }

    @staticmethod
    def _compute_load_balance(load_info: Any) -> dict[str, float]:
        """Aggregate per-rank token load into avg/max gauges."""
        if load_info is None or not torch.is_tensor(load_info) or load_info.numel() == 0:
            return {"avg_tokens": 0.0, "max_tokens": 0.0}

        tensor = load_info.detach().float()
        # Ascend moe_load is typically [layers, ranks, experts] (or 4-D multi-stage).
        if tensor.ndim >= 2:
            reduce_dims = tuple(i for i in range(tensor.ndim) if i != 1)
            per_rank = tensor.sum(dim=reduce_dims) if reduce_dims else tensor
        else:
            per_rank = tensor
        avg_tokens = float(per_rank.mean().item())
        max_tokens = float(per_rank.max().item())
        return {
            "avg_tokens": avg_tokens,
            "max_tokens": max_tokens,
        }

    # TODO: Here only expert weight exchange is considered, need to be extended to cover other weight update cases
    def compose_expert_update_info_greedy(self, updated_expert_maps, current_expert_maps):
        num_layers = current_expert_maps.shape[0]
        for layer_id in range(num_layers):
            updated_expert_maps_this_layer = updated_expert_maps[layer_id]
            current_expert_maps_this_layer = current_expert_maps[layer_id]

            expert_send_info_this_layer: dict[Any, Any] = {}
            expert_recv_info_this_layer: dict[Any, Any] = {}

            # Guard Clause: if there is no expert weight update, avoid subsequent processing
            if torch.equal(updated_expert_maps_this_layer, current_expert_maps_this_layer):
                yield (
                    expert_send_info_this_layer,
                    expert_recv_info_this_layer,
                    updated_expert_maps_this_layer,
                    layer_id,
                )
                continue

            # Parse expert_ids each rank needs to receive from other ranks
            dst_rank_indices, experts_to_recv = torch.where(
                (current_expert_maps_this_layer == -1) & (updated_expert_maps_this_layer != -1)
            )

            # Parse expert_ids each rank needs to send to other ranks
            src_rank_indices, experts_to_send = torch.where(
                (current_expert_maps_this_layer != -1) & (updated_expert_maps_this_layer == -1)
            )

            for idx in range(len(dst_rank_indices)):
                dst_rank_id = dst_rank_indices[idx].item()
                expert_id = experts_to_recv[idx].item()
                if dst_rank_id not in expert_recv_info_this_layer:
                    expert_recv_info_this_layer[dst_rank_id] = []

                if not torch.isin(torch.tensor(expert_id), experts_to_send).any():
                    # if expert_id are not sent out from any npu, it will be copied from one npu holding this expert
                    candidate_src_rank_indices = torch.where(current_expert_maps_this_layer[:, expert_id] != -1)[0]
                else:
                    candidate_src_rank_indices = src_rank_indices[experts_to_send == expert_id]

                # TODO: improve selection criterion of NPU sending expert_id,
                # considering intra-node or inter-node...
                src_rank_id = candidate_src_rank_indices[0].item()
                if src_rank_id not in expert_send_info_this_layer:
                    expert_send_info_this_layer[src_rank_id] = []

                expert_send_info_this_layer[src_rank_id].append((dst_rank_id, expert_id))
                expert_recv_info_this_layer[dst_rank_id].append((src_rank_id, expert_id))

            yield (
                expert_send_info_this_layer,
                expert_recv_info_this_layer,
                updated_expert_maps_this_layer,
                layer_id,
            )

    def calculate_rebalance_experts(self, load_info, old_placement):
        """
        Compute `new_map` by calling the `rebalance_experts` method of the policy instance.
        """
        if self.old_expert_maps is None:
            return False, None, None

        changed, priority, new_map = self.policy.rebalance_experts(old_placement, load_info)
        return changed, priority, new_map

    def get_init_expert_maps(self):
        """
        Read the initial expert_map from shared_dict.
        """
        return self.shared_dict.get("expert_maps", None)

    def fetch_and_sum_load_info(self):
        """
        Each time the subprocess is awakened, read the latest moe_load
        (shape: [num_moe_layers, num_experts_per_layer]) from shared_dict.
        """
        return self.shared_dict.get("moe_load", None)

    def update_expert_map(self, expert_maps):
        self.shared_dict["expert_maps"] = expert_maps

    def global2local(self, placement: torch.Tensor, E_local: int) -> tuple[torch.Tensor, torch.Tensor]:
        L, G, _ = placement.shape
        device = placement.device

        pt_local = torch.full((L, G, E_local), fill_value=-1, dtype=torch.long, device=device)

        valid = placement >= 0
        l_idx, g_idx, k_idx = valid.nonzero(as_tuple=True)

        slot_idx = placement[l_idx, g_idx, k_idx]

        pt_local[l_idx, g_idx, slot_idx] = k_idx

        return pt_local

    def local2global(self, placement_local: torch.Tensor) -> torch.Tensor:
        L, G, E_local = placement_local.shape
        device = placement_local.device

        max_id = torch.max(placement_local)
        E_global = (max_id + 1).item() if max_id >= 0 else 0

        if E_global == 0:
            return torch.empty((L, G, 0), dtype=torch.long, device=device)

        placement_global = torch.full((L, G, E_global), fill_value=-1, dtype=torch.long, device=device)

        valid = placement_local >= 0
        l_idx, g_idx, slot_idx = valid.nonzero(as_tuple=True)
        gid_idx = placement_local[l_idx, g_idx, slot_idx]

        placement_global[l_idx, g_idx, gid_idx] = slot_idx

        return placement_global

    def pack_update_info(self, update_info_generator):
        """
        Pack a list of update info tuples for efficient IPC.
        """
        send_all = []
        recv_all = []
        maps = []
        log2phy_all = []
        layer_ids = []

        for send_info, recv_info, new_expert_map, layer_id in update_info_generator:
            send_info_this_rank = send_info.get(self.rank_id, [])
            recv_info_this_rank = recv_info.get(self.rank_id, [])
            send_all.append(send_info_this_rank)
            recv_all.append(recv_info_this_rank)

            maps.append(new_expert_map[self.rank_id].numpy().tolist())

            log2phy_map = generate_log2phy_map(
                new_expert_map,
                self.rank_id,
                tp_size=self.tp_size,
            )
            log2phy_all.append(log2phy_map.numpy().tolist())

            layer_ids.append(layer_id)

        return list(zip(send_all, recv_all, maps, log2phy_all, layer_ids))

    @staticmethod
    def _compute_imbalance(deployment_all_layer, hotness_all_layer: np.ndarray, return_list: bool = False):
        imbalance_list = []
        deployment_all_layer = np.array(deployment_all_layer)
        for deployment, hotness in zip(deployment_all_layer, hotness_all_layer):
            counts = np.bincount(deployment.reshape(-1), minlength=hotness.shape[0])

            unit_hotness = np.divide(hotness, counts, out=np.zeros_like(hotness, dtype=float), where=counts != 0)

            stage_load = unit_hotness[deployment].sum(-1)
            stage_par = stage_load.max() / stage_load.mean()
            imbalance_list.append(stage_par)

        max_val = max(imbalance_list)
        mean_val = sum(imbalance_list) / len(imbalance_list)
        # ms-service-metric begin: optionally expose per-layer imbalance without recomputing it.
        if return_list:
            return mean_val, max_val, imbalance_list
        # ms-service-metric end.
        return mean_val, max_val

    @staticmethod
    def _calculate_hotness(deployment_all_layer, moe_load_all_layer):
        hotnesses = []
        num_of_expert = deployment_all_layer.shape[1] * deployment_all_layer.shape[2]
        for deployment, rank_load in zip(deployment_all_layer, moe_load_all_layer.numpy()):
            hotness = np.zeros(num_of_expert, dtype=rank_load.dtype)
            deployment_flat = deployment.ravel()
            rank_load_flat = rank_load.ravel()
            np.add.at(hotness, deployment_flat, rank_load_flat)
            hotnesses.append(hotness)

        return np.array(hotnesses)


class EplbProcess:
    def __init__(
        self,
        shared_dict,
        policy_type: int = 0,
        enable_d2d: bool = True,
        tp_size: int | None = None,
    ):
        """
        Args:
            shared_dict: Cross-process shared dict returned by Manager().dict()
            policy_type: Integer passed to PolicyFactory.generate_policy
            enable_d2d: Whether to enable D2D loading
        """
        self.shared_dict = shared_dict
        self.policy_type = policy_type
        self.enable_d2d = enable_d2d
        self.planner_q: Queue[Any] = Queue()
        self.block_update_q: Queue[Any] = Queue(maxsize=1)

        # Create EplbWorker instance
        self.worker = EplbWorker(
            self.shared_dict,
            self.policy_type,
            self.enable_d2d,
            tp_size=tp_size,
        )

    def worker_process(self, planner_q, block_update_q):
        """
        Subprocess entry: bind to specified NPU, loop waiting for planner_q to wake up,
        call do_update, then notify main process update is complete.
        """
        try:
            from ms_service_metric.adapters.vllm.adapter import get_vllm_adapter, initialize_vllm_metric  # type: ignore

            initialize_vllm_metric()
            adapter = get_vllm_adapter()
            logger.info("[EPLB metrics] The adapter initialized: %s", adapter.is_initialized())
        except Exception as e:
            logger.warning("[EPLB metrics] Failed to initialize metrics: %s", e)

        if self.policy_type == 3:
            from vllm_ascend.eplb.core.policy.policy_flashlb import warm_up

            warm_up()
        while True:
            try:
                planner_q.get()

                packed_update_info = self.worker.do_update()

                while True:
                    if not block_update_q.empty():
                        continue
                    block_update_q.put(packed_update_info)
                    break

            except Exception as e:
                logger.warning(
                    "[eplb/worker] Subprocess crashed, EPLB optimization will stop. error=%s",
                    e,
                    exc_info=True,
                )
                break

    def _launch_process(self):
        """
        Use spawn method to launch subprocess and return (planner_q, block_update_q, proc).
        """
        proc = Process(target=self.worker_process, args=(self.planner_q, self.block_update_q), daemon=True)

        proc.start()
        return proc
