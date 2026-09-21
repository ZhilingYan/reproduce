# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from typing import List
import re


def textcraft_projection(actions: List[str]):
    """Parse model outputs into TextCraft env actions.

    Expected format (mirrors webshop / alfworld):
        <think>some reasoning...</think><action>craft ... / get N ... / inventory</action>

    The content between ``<action>`` and ``</action>`` is extracted and passed
    to ``TextCraft.step``. Minecraft item ids are lowercase with spaces, so we
    lowercase the extracted action (safe for ``craft``/``get``/``inventory`` and
    for item names). Validity requires a well-formed ``<action>`` block AND a
    ``<think>`` block, and rejects outputs containing Chinese characters.

    Returns ``(actions, valids)`` where ``actions`` is the list of extracted
    action strings (in place) and ``valids`` is a 0/1 list.
    """
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original for <think> check
        lowered = actions[i].lower()

        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = lowered.find(start_tag)
        end_idx = lowered.find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
                # No valid <action>...</action> block -> invalid, keep a short
                # tail so env.step gets a (failing) string rather than garbage.
                actions[i] = lowered[-30:]
                continue

            extracted_action = lowered[start_idx + len(start_tag):end_idx].strip()
            actions[i] = extracted_action
            valids[i] = 1
        except Exception:
            actions[i] = lowered[-30:]

        # require a <think>...</think> block
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # reject Chinese characters
        if re.search(r"[一-鿿]", original_str):
            valids[i] = 0

    return actions, valids
