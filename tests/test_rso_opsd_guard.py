# -*- coding: utf-8 -*-
"""RSO+OPSD 的守卫测试:克隆保真 / 参数接线 / 断言反转 / 既有入口零改动。

对应 RSO_method_design §2a"入口与守卫"条款的两条守卫测试(checklist #1):
  ① 克隆 fit 与源 fit 逐行 diff,差异只允许 OPSD 插入块;
  ② 参数存在性断言——config 值 == trainer 属性,且未知键(拼写错误)拒绝启动。
外加底线:main_rso / rso_core / sdar_utils / recursive_factory 未被牵连,
dp_actor 与编排器/收集器的加法全部缺省关闭。

运行(不需要 GPU):PYTHONPATH=$PWD python tests/test_rso_opsd_guard.py
"""
from __future__ import annotations

import difflib
import inspect
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0


def check(cond, msg):
    global passed
    assert cond, f"✗ {msg}"
    passed += 1
    print(f"  ✓ {msg}")


def _norm(src: str):
    """去掉注释与空行后逐行比较(克隆刻意省略了来源里的读码笔记注释,代码行必须逐字保真)。
    fit 正文的字符串字面量里没有 '#',朴素切分安全(有的话本测试会先在等式侧失败)。"""
    out = []
    for line in src.splitlines():
        code = line.split("#", 1)[0].rstrip()
        if code.strip():
            out.append(code)
    return out


# ---------------------------------------------------------------------------- G1
def test_G1_fit_clone_only_opsd_insertions():
    print("G1 守卫①:克隆 fit 相对 RayPPOTrainer.fit(阶段 1 RSO 的 fit)只有 OPSD 插入")
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.trainer.ppo.rso_opsd_ray_trainer import RSOOPSDRayTrainer

    theirs = _norm(inspect.getsource(RayPPOTrainer.fit))
    ours = _norm(inspect.getsource(RSOOPSDRayTrainer.fit))
    sm = difflib.SequenceMatcher(None, theirs, ours, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    kinds = sorted({op[0] for op in ops})
    check(kinds in ([], ["insert"]), f"只有插入,来源代码行零删改(实际操作类型 {kinds}:{ops[:3]})")
    check(len(ops) == 1, f"插入集中在恰好一个连续块(实际 {len(ops)} 处)")
    inserted = "\n".join("\n".join(ours[op[3]:op[4]]) for op in ops)
    for kw in ["teacher_forward", "teacher_log_probs", "act_mask",
               "build_action_token_mask", "_compute_teacher_log_probs"]:
        check(kw in inserted, f"插入块包含 {kw}")
    check("compute_advantage" not in inserted and "rso_progress_coef" not in inserted,
          "插入块不碰优势计算——rso_* 参数通路来自克隆源本身,无需也不许重复(P6 型漏搬的反面)")


# ---------------------------------------------------------------------------- G2
def _mk_cfg(**overrides):
    from omegaconf import OmegaConf
    base = {
        "algorithm": {
            "adv_estimator": "rso",
            "filter_groups": {"enable": False},
            "rso": {"progress_coef": 0.1, "progress_clip": 3, "progress_baseline_loo": True,
                    "invalid_coef": 0.1},
            "rso_opsd": {"gate_beta": 2.5, "lambda_coef": 0.01},
        },
        "actor_rollout_ref": {
            "actor": {"entropy_coeff": 0, "use_kl_loss": True, "kl_loss_coef": 0.01,
                      "kl_loss_type": "low_var_kl", "use_invalid_action_penalty": False,
                      "use_rso_opsd_loss": True, "rso_opsd_loss_coef": 0.01,
                      "rso_opsd_gate_beta": 2.5},
            "rollout": {"n": 1},
        },
        "env": {"env_name": "textcraft_synth"},
        "data": {"val_batch_size": 50},
    }
    cfg = OmegaConf.create(base)
    from omegaconf import OmegaConf as OC
    for path, v in overrides.items():
        OC.update(cfg, path, v, force_add=True)
    return cfg


def test_G2_config_asserts_kl_flipped():
    print("G2 断言反转:OPSD 入口要求 KL 开 0.01 low_var_kl;熵仍必须 0(各守各的)")
    from verl.trainer.main_rso_opsd import validate_rso_opsd_config
    validate_rso_opsd_config(_mk_cfg())
    check(True, "合规配置(KL 开 0.01 / 熵 0 / rso estimator)通过")
    for path, val, label in [
        ("actor_rollout_ref.actor.use_kl_loss", False, "KL 关被拒(§一点五)"),
        ("actor_rollout_ref.actor.kl_loss_coef", 0.001, "kl_loss_coef≠0.01 被拒"),
        ("actor_rollout_ref.actor.kl_loss_type", "kl", "kl_loss_type≠low_var_kl 被拒"),
        ("actor_rollout_ref.actor.entropy_coeff", 0.001, "熵奖励被拒(崩塌教训不豁免)"),
        ("algorithm.rso_opsd.lambda_coef", 0.0, "λ=0 被拒(蒸馏等于没开)"),
        ("algorithm.adv_estimator", "grpo", "优势必须仍是 rso"),
    ]:
        try:
            validate_rso_opsd_config(_mk_cfg(**{path: val}))
            check(False, label + "(没拒!)")
        except AssertionError:
            check(True, label)


def test_G2b_wiring_guard():
    print("G2b 守卫②:config 值==trainer 属性;未知键(拼写错误)拒绝启动")
    from verl.trainer.main_rso_opsd import validate_rso_opsd_wiring
    trainer = SimpleNamespace(
        rso_params={"progress_coef": 0.1, "progress_clip": 3.0, "progress_baseline_loo": True,
                    "invalid_coef": 0.1, "invalid_gate_min_valid_ratio": 0.0},
        opsd_gate_beta=2.5, opsd_lambda=0.01)
    validate_rso_opsd_wiring(trainer, _mk_cfg())
    check(True, "接线一致时通过")
    try:
        validate_rso_opsd_wiring(trainer, _mk_cfg(**{"algorithm.rso.progres_coef": 0.2}))
        check(False, "未知键 progres_coef 应被拒")
    except AssertionError:
        check(True, "algorithm.rso 的拼写错误键 → 启动失败(不再静默用默认值跑)")
    bad_trainer = SimpleNamespace(rso_params=dict(trainer.rso_params, progress_coef=0.5),
                                  opsd_gate_beta=2.5, opsd_lambda=0.01)
    try:
        validate_rso_opsd_wiring(bad_trainer, _mk_cfg())
        check(False, "config 与 trainer 属性不一致应被拒")
    except AssertionError:
        check(True, "config 值未到达 trainer → 启动失败(值相等检查,非键存在检查)")


# ---------------------------------------------------------------------------- G3
def test_G3_stage1_entries_untouched():
    print("G3 底线:阶段 1 与 flat 的入口/内核零牵连")
    from verl.trainer import main_rso
    from verl.trainer.ppo import rso_core, sdar_utils
    from agent_system.environments.env_package.textcraft_synth import recursive_factory

    src = inspect.getsource(main_rso)
    check("rso_opsd" not in src, "main_rso 里没有任何 OPSD 的影子")
    check("use_kl_loss=False" in src.replace(" ", "") or "use_kl_loss" in src,
          "main_rso 的 KL 关断言原文仍在")
    check("assert not bool(config.actor_rollout_ref.actor.get(\"use_kl_loss\", False))"
          in src, "main_rso 仍然硬断言 KL 关(永久,例外条款)")
    check("teacher" not in inspect.getsource(rso_core).lower(), "rso_core 无老师相关内容")
    check("sigmoid" in inspect.getsource(sdar_utils.compute_sdar_loss),
          "flat 在用的 compute_sdar_loss 未被改动(仍是 sigmoid 门)")
    check("opsd" not in inspect.getsource(recursive_factory).lower(),
          "阶段 1 环境工厂未被改动")

    from verl.workers.actor import dp_actor
    dp_src = inspect.getsource(dp_actor)
    n_gate = dp_src.count('self.config.get("use_rso_opsd_loss", False)')
    check(n_gate == 2, f"dp_actor 的两处加法(select_keys/损失)都在独立开关后面(实际 {n_gate} 处)")
    check('get("use_sdar_loss", False)' in dp_src, "use_sdar_loss 分支原样保留")

    from agent_system.recursive import orchestrator
    orch_src = inspect.getsource(orchestrator)
    check(orch_src.count("_priv_enabled") >= 5, "编排器的 priv 逻辑全部躲在 _priv_enabled 后")
    check('callable(getattr(adapter, "build_priv", None))' in orch_src,
          "priv 通道由适配器能力探测决定(阶段 1 适配器无 build_priv → 恒关)")

    from agent_system.multi_turn_rollout import recursive_rollout_loop
    check('if "node_priv" in meta:' in inspect.getsource(recursive_rollout_loop),
          "收集器只在 turn_meta 带键时回填新列(RSO 批不多列)")


# ---------------------------------------------------------------------------- G4
def test_G4_run_script_differs_only_by_opsd_lines():
    print("G4 训练脚本相对 RSO 8 卡脚本只差 OPSD 行(入口/rso_opsd 参数/KL 三行/val50/输出名)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    a = [l for l in open(os.path.join(root, "examples/rso_8gpu/run_synth_rso_8gpu.sh"),
                         encoding="utf-8") if not l.lstrip().startswith("#")]
    b = [l for l in open(os.path.join(root, "examples/rso_8gpu/run_synth_rso_opsd_8gpu.sh"),
                         encoding="utf-8") if not l.lstrip().startswith("#")]
    only_b = [l.strip() for l in b if l not in a]
    only_a = [l.strip() for l in a if l not in b]

    def _allowed_new(l):
        return any(t in l for t in ("main_rso_opsd", "algorithm.rso_opsd.", "use_kl_loss=True",
                                    "kl_loss_coef=0.01", "kl_loss_type=low_var_kl",
                                    "val_batch_size=50", "rso_runs/rso_opsd",
                                    "experiment_name='synth_rso_opsd_8gpu'"))

    def _allowed_gone(l):
        return any(t in l for t in ("main_rso ", "main_rso \\", "use_kl_loss=False",
                                    "val_batch_size=100", "rso_runs/rso",
                                    "experiment_name='synth_rso_8gpu'"))

    check(all(_allowed_new(l) for l in only_b), f"新增行全部在允许清单内: {only_b}")
    check(all(_allowed_gone(l) for l in only_a), f"减少行全部在允许清单内: {only_a}")
    check(any("gate_beta=2.5" in l for l in only_b), "β=2.5(§2c 初始值)")
    check(any("lambda_coef=0.01" in l for l in only_b), "λ=0.01(§2c 初始值)")
    check(any("val_batch_size=50" in l for l in only_b), "val_batch_size=50(checklist #4)")


if __name__ == "__main__":
    for fn in [test_G1_fit_clone_only_opsd_insertions,
               test_G2_config_asserts_kl_flipped,
               test_G2b_wiring_guard,
               test_G3_stage1_entries_untouched,
               test_G4_run_script_differs_only_by_opsd_lines]:
        fn()
    print(f"\n全部通过: {passed} 条断言")
