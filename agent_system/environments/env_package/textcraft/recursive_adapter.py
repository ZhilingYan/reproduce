# -*- coding: utf-8 -*-
"""真 TextCraft(ADaPT 官方环境)的递归适配器:synth 适配器的子类,五处覆写。

出处:scripts_rso/eval_textcraft_ood.py 的 RealTextcraftAdapter 迁入 env_package,
供 repo 正式 val_only 流程(main_rso 的 textcraft 分支)使用。覆写清单:
  ① filler_action:官方 env 没有无副作用 noop,用 RECURSIVE_NOOP 哨兵,
     worker.step 对它短路(envs.py);
  ② on_node_open:删 PhiTracker(推理不需要 Φ,且 synth 配方库不识真实物品),
     root 记录配方清单,子节点经 env_info 的 root_node_uid 精确继承清单;
  ③ parse_delegation:目标解析正则放宽到带空格的真实物品名
     (synth 物品名是 a1_i5 式无空格,按 and/逗号切;真实名如 dark oak planks
     含空格,只按逗号与 craft 词切段);
  ④ build_observation:配方清单上榜进任务区,模板换 OOD 派生版(prompts/textcraft_ood.py);
  ⑤ _state_block:known_recipes 段永空(配方已上榜),只保留库存快照段。
判定(_satisfied 库存净增量)、按节点卡死检测、战报、预算拒绝等全部继承 synth 版
——库存字典语义两环境相同(键 = 空格式物品名)。
"""
import re

from agent_system.environments.env_package.textcraft.envs import RECURSIVE_NOOP
from agent_system.environments.env_package.textcraft_synth.recursive_adapter import (
    TextCraftSynthRecursiveAdapter, _DELEGATE_PREFIX_RE, _USING_CLAUSE_RE)
from agent_system.environments.prompts.textcraft_ood import (
    TEXTCRAFT_OOD_REC_CHILD_TEMPLATE, TEXTCRAFT_OOD_REC_CHILD_TEMPLATE_NO_HIS,
    TEXTCRAFT_OOD_REC_TEMPLATE, TEXTCRAFT_OOD_REC_TEMPLATE_NO_HIS)

# 真实物品名可含空格:不再按 and 切段,craft 词与逗号做分隔
_TARGET_RE_SP = re.compile(r"^\s*(\d+)\s*x?\s+([a-z0-9_][a-z0-9_ ]*?)[\s\.;:!]*$", re.IGNORECASE)
_SEGMENT_SPLIT_SP = re.compile(r",|\bcraft\b", re.IGNORECASE)
_CRAFT_TOKEN_SP = re.compile(r"\bcraft\s+(\d+)\s*x?\s+([a-z0-9_][a-z0-9_ ]*?)\s*(?:,|\||$)", re.IGNORECASE)


class RealTextcraftAdapter(TextCraftSynthRecursiveAdapter):

    def __init__(self, config=None, **kw):
        super().__init__(config, **kw)
        self._commands = {}                              # root uid → 配方清单文本

    def filler_action(self) -> str:
        return RECURSIVE_NOOP

    def on_node_open(self, node, env_info):
        # 照抄父类快照/诊断段;唯一删除:PhiTracker
        node.open_snapshot = {"inventory": dict(env_info.get("extra.inventory", {}))}
        node.scratch.setdefault("context", "")
        node.scratch["known_recipes"] = {}
        node.scratch["inv_snapshot"] = None
        node.scratch["inv_step"] = None
        node.scratch["preexisting"] = (not node.is_root) and self._satisfied(node, env_info)
        if node.is_root:
            cmds = env_info.get("extra.commands") or []
            self._commands[node.uid] = "\n".join(cmds)
        else:
            # 多槽并行:按 root_node_uid 精确继承本树的清单,避免串树
            root_uid = str(env_info.get("root_node_uid") or "")
            if root_uid in self._commands:
                node.scratch["commands_text"] = self._commands[root_uid]

    def parse_delegation(self, action_text):
        text = (action_text or "").strip()
        pm = _DELEGATE_PREFIX_RE.match(text)
        if pm is None:
            return None
        from agent_system.recursive.protocol import DelegationRequest
        body = text[pm.end():]
        head, sep, tail = body.partition("|")
        context = re.sub(r"\s+", " ", tail).strip(" ,;.") if sep else ""
        head = _USING_CLAUSE_RE.sub(" ", head)
        goal = {}
        segs = [s for s in _SEGMENT_SPLIT_SP.split(head) if s.strip()]
        if not segs:
            return None
        for s in segs:
            tm = _TARGET_RE_SP.match(s.strip())
            if tm is None:
                return None
            n, item = int(tm.group(1)), re.sub(r"\s+", " ", tm.group(2)).strip().lower()
            if n <= 0:
                return None
            goal[item] = goal.get(item, 0) + n
        for cm in _CRAFT_TOKEN_SP.finditer(tail):
            n, item = int(cm.group(1)), re.sub(r"\s+", " ", cm.group(2)).strip().lower()
            if n > 0 and item not in goal:
                goal[item] = n
        return DelegationRequest(goal=goal, context=context, raw_action=text)

    def _task_lines(self, node):
        base = super()._task_lines(node)
        cmds = self._commands.get(node.uid) or node.scratch.get("commands_text", "")
        return base + ("\n\nCrafting commands:\n" + cmds if cmds else "")

    def build_observation(self, node, is_first_turn):
        tpl_no_his = (TEXTCRAFT_OOD_REC_TEMPLATE_NO_HIS if node.is_root
                      else TEXTCRAFT_OOD_REC_CHILD_TEMPLATE_NO_HIS)
        tpl_his = (TEXTCRAFT_OOD_REC_TEMPLATE if node.is_root
                   else TEXTCRAFT_OOD_REC_CHILD_TEMPLATE)
        task = self._task_lines(node)
        if is_first_turn or not node.history:
            return tpl_no_his.format(current_observation=task)
        current = task + "\n\nResult of your last action: " + node.last_result + self._state_block(node)
        if self.history_length <= 0:
            return tpl_no_his.format(current_observation=current)
        H = self.history_length
        hist = node.history
        recent = hist[-H:]
        start = len(hist) - len(recent)
        lines = []
        for j, turn in enumerate(recent):
            k = start + j
            obs_before = task if k == 0 else hist[k - 1].result
            lines.append(f"Step {k + 1}:{turn.action} {obs_before}\n")
        return tpl_his.format(current_observation=current, step_count=len(hist),
                              history_length=len(recent), action_history="\n".join(lines),
                              current_step=len(hist) + 1)

    def _state_block(self, node):
        # 配方上榜,known_recipes 段永远为空;只保留库存快照段
        lines = []
        if node.scratch.get("inv_snapshot") is not None:
            lines.append(f"Your inventory as of step {node.scratch['inv_step']} "
                         f"(when you last checked): {node.scratch['inv_snapshot']}")
        return ("\n" + "\n".join(lines)) if lines else ""
