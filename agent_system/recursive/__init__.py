# -*- coding: utf-8 -*-
"""RAO(Recursive Agent Optimization, arXiv:2605.06639)的递归执行内核。

这个包是 RAO 官方 platoon 框架 episode 层在 lockstep 架构下的等价实现,
**与具体环境无关**:不 import torch、不 import verl、不认识 TextCraft。
想让某个 benchmark 支持递归委托,只需实现 protocol.py 里的 RecursiveEnvAdapter。

移植来源(路径相对 Experiment/code_references/platoon/):
  platoon/episode/trajectory.py     节点数据结构与预算语义
  platoon/episode/loop.py           单 agent 的执行循环
  platoon/agents/actions/subagent.py 委托动作的时序

设计文档:docs/RAO_PORT_DESIGN.md(§9ter 有逐文件的源码行号对照表)。
"""
