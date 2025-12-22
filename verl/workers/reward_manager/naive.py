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


def _process_single_item_naive(args):
    """Process single data item for multiprocessing pool"""
    i, data_item, tokenizer, compute_score, reward_fn_key = args
    
    try:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        # ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        ground_truth = data_item.non_tensor_batch["extra_info"]["answer"]
        data_source = data_item.non_tensor_batch[reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["num_turns"] = num_turns

        score = compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

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
        print(f"Error processing data item {i}: {str(e)}")
        return None


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager with multiprocessing support."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", 
                 max_workers=None) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            max_workers: Maximum number of worker processes. If None, uses CPU count.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.max_workers = 10 # max_workers or multiprocessing.cpu_count()
        # print("max_workers:" + str(self.max_workers))
        # print("*************************")

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Process data and compute rewards with multiprocessing"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
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

        # Prepare tasks for parallel processing
        tasks = [
            (i, data[i], self.tokenizer, self.compute_score, self.reward_fn_key)
            for i in range(len(data))
        ]

        # Process in parallel using multiprocessing pool
        with Pool(processes=self.max_workers) as pool:
            for result in pool.imap(_process_single_item_naive, tasks, chunksize=4):
                if result is None:
                    continue

                # Update reward tensor
                i = result["index"]
                valid_response_length = result["valid_response_length"]
                score = result["score"]
                
                # Handle score result
                if isinstance(score, dict):
                    reward = score["score"]
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score
                    
                reward_tensor[i, valid_response_length - 1] = reward

                # Control debug printing
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
