# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from typing import Any
import multiprocessing
from multiprocessing import Pool
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _process_single_item(args):
    """处理单个数据项的函数，用于进程池调用"""
    i, data_item, tokenizer, compute_score, reward_fn_key = args
    
    try:
        # 提取数据项信息
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # 解码
        prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch[reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["num_turns"] = num_turns

        # 计算分数 - 这是CPU密集型任务，并行处理的核心
        score = compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        # 返回处理结果
        return {
            "index": i,
            "valid_response_length": valid_response_length,
            "score": score,
            "prompt_str": prompt_str,
            "response_str": response_str,
            "ground_truth": ground_truth,
            "data_source": data_source
        }
    except Exception as e:
        print(f"处理数据项 {i} 时出错: {str(e)}")
        return None


@register("scad")
class ScadRewardManager(AbstractRewardManager):
    """The reward manager with parallel processing support."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", 
                 max_workers=None) -> None:
        """
        初始化奖励管理器，支持并行处理

        Args:
            tokenizer: 用于解码的tokenizer
            num_examine: 用于调试打印的样本数量
            compute_score: 计算分数的函数，默认为default_compute_score
            reward_fn_key: 获取数据源的键名
            max_workers: 最大进程数，None则使用CPU核心数
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        # 设置进程池大小，默认使用CPU核心数
        self.max_workers = max_workers or multiprocessing.cpu_count()

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """处理数据并计算奖励，使用多进程并行处理compute_score"""

        # 如果已有rm_scores，直接返回
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = defaultdict(int)

        # 准备并行任务参数
        tasks = [
            (i, data[i], self.tokenizer, self.compute_score, self.reward_fn_key)
            for i in range(len(data))
        ]

        # 使用进程池并行处理
        with Pool(processes=self.max_workers) as pool:
            # 处理结果，保持原始顺序
            for result in pool.imap(_process_single_item, tasks):
                if result is None:
                    continue

                # 更新奖励张量
                i = result["index"]
                valid_response_length = result["valid_response_length"]
                score = result["score"]
                
                # 处理分数结果
                if isinstance(score, dict):
                    reward = score["score"]
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score
                    
                reward_tensor[i, valid_response_length - 1] = reward

                # 控制打印信息
                data_source = result["data_source"]
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", result["prompt_str"])
                    print("[response]", result["response_str"])
                    print("[ground_truth]", result["ground_truth"])
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
    