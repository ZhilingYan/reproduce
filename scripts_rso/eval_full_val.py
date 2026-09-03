#!/usr/bin/env python3
"""训练完成后的全量验证集评测。

用途:拿训练好的 checkpoint(或未训练的基座模型),在 TextCraft-Synth 的
**全量 632 道验证题**上逐题跑推理,产出两样东西:
  1) 汇总指标(总体 / 分难度 / easy+medium 合并口径),写 <out>_metrics.json;
  2) 每道题的完整结果(逐轮 prompt、模型输出、解析动作、环境反馈、
     步数、reward、是否成功、gold 计划),写 <out>_cases.jsonl —— 每行一题。

为什么单独写一个脚本而不复用训练时的验证:训练时的验证走 lockstep 批处理,
验证环境进程数 = val_batch_size,用全量 632 会常驻 632 个 Ray worker 而 OOM
(所以训练里用的是 val100 固定子集)。本脚本改为「顺序单环境 + vLLM 批量生成」,
内存恒定,可以安心跑全量,且能把每题的完整轨迹留档。

依赖:vLLM(直接加载权重,不需要起服务)。

用法:
  # 评测训练后的 checkpoint
  python scripts_rso/eval_full_val.py \
      --model $HOME/rso_runs/qwen_grpo/ckpts/global_step_150/actor/huggingface \
      --out   $HOME/rso_runs/qwen_grpo/eval_full

  # 评测基座模型(训练前基线)
  python scripts_rso/eval_full_val.py --model Qwen/Qwen3.5-4B --out ./eval_base

  # 只测部分难度 / 限制题数(调试用)
  python scripts_rso/eval_full_val.py --model <M> --out <O> \
      --difficulties easy medium --limit 20
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_system.environments.env_package.textcraft_synth.synth_core import (  # noqa: E402
    SynthTextCraftEnv, load_tasks, get_shared_recipe_db)
from agent_system.environments.env_package.textcraft_synth.projection import (  # noqa: E402
    textcraft_synth_projection)
from agent_system.environments.prompts.textcraft_synth import (  # noqa: E402
    TEXTCRAFT_SYNTH_TEMPLATE, TEXTCRAFT_SYNTH_TEMPLATE_NO_HIS)


def build_prompt(task_obs, last_result, history, history_length, step_idx):
    """与训练时 TextCraftSynthEnvironmentManager.build_text_obs 完全一致的拼接。

    每轮 prompt = 任务描述 + 上一步动作结果(含状态块) + 最近 N 步滑窗。
    """
    if step_idx == 0 or history_length <= 0 or not history:
        return TEXTCRAFT_SYNTH_TEMPLATE_NO_HIS.format(current_observation=task_obs)
    recent = history[-history_length:]
    lines = []
    start = len(history) - len(recent)
    for k, h in enumerate(recent):
        lines.append(f"[Observation {start + k + 1}: '{h['obs']}', "
                     f"Action {start + k + 1}: '{h['action']}']")
    return TEXTCRAFT_SYNTH_TEMPLATE.format(
        current_observation=task_obs + "\n\nResult of your last action: " + last_result,
        step_count=len(history),
        history_length=len(recent),
        action_history="\n".join(lines),
        current_step=len(history) + 1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF 模型名或本地 checkpoint 目录(actor/huggingface)")
    ap.add_argument("--out", required=True, help="输出前缀,会生成 _metrics.json 与 _cases.jsonl")
    ap.add_argument("--split", default="val", help="val=全量632题; val100=固定子集")
    ap.add_argument("--difficulties", nargs="+",
                    default=["easy", "medium", "hard", "extreme"])
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 题(调试)")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑:跳过 <out>_cases.jsonl 里已完成的题,结果追加写入")
    ap.add_argument("--task-ids", nargs="+", default=None,
                    help="只跑指定 task_id(调试单题用),如 textcraft_synth.val.74")
    ap.add_argument("--max-steps", type=int, default=2000,
                    help="每题的 episode 步数上限。默认 2000 是正式评测口径:"
                         "深题的 gold 最多 209 步,2000 给足余量,确保失败反映能力而非预算")
    ap.add_argument("--history-length", type=int, default=2, help="与训练一致的滑窗长度")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="默认 0(贪心解码),与 RAO 官方推理协议一致,结果可复现")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="同时推进的题数(越大越快,显存换速度)")
    ap.add_argument("--tp", type=int, default=1, help="vLLM 张量并行度(9B 建议 2-4)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--save-traj", action="store_true", default=True,
                    help="在 _cases.jsonl 里保存逐轮完整轨迹(默认开)")
    ap.add_argument("--no-save-traj", dest="save_traj", action="store_false")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tasks = load_tasks(args.split, args.difficulties)
    if args.task_ids:
        want = set(args.task_ids)
        tasks = [t for t in tasks if t.get("id") in want]
        if not tasks:
            raise SystemExit(f"没有匹配的 task_id: {args.task_ids}")
    if args.limit > 0:
        tasks = tasks[:args.limit]

    # 断点续跑:已完成的题直接跳过,其汇总记录也一并载入,最终指标仍覆盖全部题
    prior_results = []
    cases_path = args.out + "_cases.jsonl"
    if args.resume and os.path.exists(cases_path):
        done_ids = set()
        for line in open(cases_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue      # 上次被杀时写了半行,丢弃
            done_ids.add(r["task_id"])
            prior_results.append({k: r.get(k, 0) for k in RESULT_KEYS})
        tasks = [t for t in tasks if t.get("id") not in done_ids]
        print(f"[eval] 续跑:已完成 {len(done_ids)} 题,跳过;剩余 {len(tasks)} 题", flush=True)
    print(f"[eval] split={args.split} 难度={args.difficulties} 题数={len(tasks)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, trust_remote_code=True,
              max_model_len=16384)
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)

    db = get_shared_recipe_db()
    out_cases = open(cases_path, "a" if args.resume else "w")
    results = list(prior_results)
    t_start = time.time()

    # 分批推进:每批 batch_size 道题同时走,批内用 vLLM 一次生成多条,吞吐高
    for b0 in range(0, len(tasks), args.batch_size):
        batch = tasks[b0:b0 + args.batch_size]
        envs, states = [], []
        for t in batch:
            e = SynthTextCraftEnv(db)
            obs, info = e.reset(t, max_steps_override=args.max_steps)
            envs.append(e)
            # prev_obs = 本轮动作发生【之前】智能体看到的观察。训练时 memory 存的
            # 就是 (动作前的观察, 该动作) 这一对(env_manager.step 里 pre_text_obs
            # 在 store 之后才更新),评测必须一致,否则滑窗内容错位一格。
            states.append({"task": t, "task_obs": obs, "last_result": "",
                           "prev_obs": obs, "history": [], "done": False,
                           "reward": 0.0, "turns": 0, "traj": [],
                           "gt_plan": info.get("extra.gt_plan", ""),
                           # token 消耗(2026-08-26 加):整条轨迹的 input / output token 累计。
                           # 训练与训练中 val 不算这个;只有本脚本(final 全量 val)算。
                           "input_tokens": 0, "output_tokens": 0})

        for step in range(args.max_steps):
            live = [i for i, s in enumerate(states) if not s["done"]]
            if not live:
                break
            prompts = []
            for i in live:
                s = states[i]
                p = build_prompt(s["task_obs"], s["last_result"], s["history"],
                                 args.history_length, s["turns"])
                prompts.append(tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True))
            outs = llm.generate(prompts, sp, use_tqdm=False)

            for k, i in enumerate(live):
                s = states[i]
                raw = outs[k].outputs[0].text
                # 精确 token 数直接取 vLLM 的返回值(它生成时用的就是这些 id),不重新 tokenize:
                # prompt_token_ids = 这一轮模型实际吃进去的完整 prompt(含 chat template);
                # outputs[0].token_ids = 这一轮生成的 token(含结束符)。
                n_in = len(outs[k].prompt_token_ids or [])
                n_out = len(outs[k].outputs[0].token_ids or [])
                s["input_tokens"] += n_in
                s["output_tokens"] += n_out
                actions, valids = textcraft_synth_projection([raw])
                obs, rew, done, info = envs[i].step(actions[0])
                s["turns"] += 1
                s["history"].append({"obs": s["prev_obs"], "action": actions[0]})
                s["prev_obs"] = obs
                s["last_result"] = obs
                if args.save_traj:
                    s["traj"].append({
                        "turn": s["turns"], "prompt": prompts[k], "output": raw,
                        "action": actions[0], "valid": int(valids[0]),
                        "env_feedback": obs, "reward": float(rew), "done": bool(done),
                        "input_tokens": n_in, "output_tokens": n_out,
                    })
                if done:
                    s["done"] = True
                    s["reward"] = float(rew)
                    flush_case(out_cases, results, s, args.save_traj)

        for s in states:          # 收尾:把跑满步数上限仍未完成的题落盘
            if not s.get("flushed"):
                flush_case(out_cases, results, s, args.save_traj)
        out_cases.flush()
        done_n = b0 + len(batch)
        acc = sum(r["success"] for r in results) / max(len(results), 1)
        print(f"[eval] {done_n}/{len(tasks)} 题完成, 当前总体成功率 {acc:.3f}, "
              f"已用 {(time.time()-t_start)/60:.1f} 分钟", flush=True)
        write_metrics(args, results, t_start)

    out_cases.close()
    write_metrics(args, results, t_start, final=True)


# 每题进汇总(results)的字段;--resume 读旧 _cases.jsonl 时缺的键(老文件没有 token 字段)补 0
RESULT_KEYS = ("task_id", "difficulty", "success", "reward", "turns_used", "gold_plan_len",
               "input_tokens", "output_tokens", "total_tokens")


def flush_case(out_cases, results, s, save_traj=True):
    """把一道题的完整结果写进 jsonl 并累计进汇总。每题一结束就调用,
    这样即便作业撞墙钟被杀,已完成的题也全部保住。"""
    if s.get("flushed"):
        return
    s["flushed"] = True
    t = s["task"]
    rec = {
        "task_id": t.get("id"),
        "goal": t.get("goal"),
        "difficulty": t["misc"].get("difficulty"),
        "max_depth": t["misc"].get("max_depth"),
        "gold_plan_len": len(t["misc"].get("gold_trajectory", [])),
        "gt_plan": s["gt_plan"],
        "success": s["reward"] == 1.0,
        "reward": s["reward"],
        "turns_used": s["turns"],
        # 整条轨迹的 token 消耗:每轮 prompt 都重新编码全部历史,input 随轮数近似平方增长,
        # 所以 input 远大于 output——这正是 RAO 论文表 9 比较 single vs recursive 的口径。
        "input_tokens": int(s.get("input_tokens", 0)),
        "output_tokens": int(s.get("output_tokens", 0)),
        "total_tokens": int(s.get("input_tokens", 0)) + int(s.get("output_tokens", 0)),
        "trajectory": s["traj"] if save_traj else None,
    }
    out_cases.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_cases.flush()
    results.append({k: rec[k] for k in RESULT_KEYS})


def _token_stats(rs):
    """一组题的 token 汇总:均值 / 中位 / 最大,以及 input 与 output 各自的均值。"""
    if not rs:
        return None
    tot = sorted(r.get("total_tokens", 0) for r in rs)
    n = len(tot)
    return {
        "n": n,
        "mean_input_tokens": sum(r.get("input_tokens", 0) for r in rs) / n,
        "mean_output_tokens": sum(r.get("output_tokens", 0) for r in rs) / n,
        "mean_total_tokens": sum(tot) / n,
        "median_total_tokens": tot[n // 2],
        "max_total_tokens": tot[-1],
        "sum_total_tokens": sum(tot),
    }



def write_metrics(args, results, t_start, final=False):
    """汇总指标。评测过程中定期落盘,作业撞墙钟也能保住已完成部分。"""
    by_diff = defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r["success"])
    em = [r["success"] for r in results if r["difficulty"] in ("easy", "medium")]
    metrics = {
        "completed": final,
        "model": args.model,
        "split": args.split,
        "difficulties": args.difficulties,
        "n_tasks": len(results),
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "overall_success_rate": sum(r["success"] for r in results) / max(len(results), 1),
        "easymedium_success_rate": (sum(em) / len(em)) if em else None,
        "per_difficulty": {d: {"n": len(v), "success_rate": sum(v) / len(v)}
                           for d, v in sorted(by_diff.items())},
        "mean_turns_used": sum(r["turns_used"] for r in results) / max(len(results), 1),
        "mean_turns_used_on_success": (
            sum(r["turns_used"] for r in results if r["success"])
            / max(sum(r["success"] for r in results), 1)),
        # token 消耗(2026-08-26 加):每题一整条轨迹的 input/output token,按 全部 / 分难度 /
        # 成功 vs 失败 汇总。数值来自 vLLM 返回的精确 token id 计数。
        "tokens": {
            "overall": _token_stats(results),
            "per_difficulty": {d: _token_stats([r for r in results if r["difficulty"] == d])
                               for d in sorted(by_diff)},
            "on_success": _token_stats([r for r in results if r["success"]]),
            "on_failure": _token_stats([r for r in results if not r["success"]]),
        },
        "wallclock_minutes": (time.time() - t_start) / 60,
        "cases_file": os.path.basename(args.out + "_cases.jsonl"),
    }
    with open(args.out + "_metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if not final:
        return

    print("\n===== 评测完成 =====")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_difficulty"},
                     ensure_ascii=False, indent=2))
    print("分难度:", json.dumps(metrics["per_difficulty"], ensure_ascii=False, indent=2))
    print(f"\n每题完整结果(含轨迹): {args.out}_cases.jsonl")
    print(f"汇总指标:             {args.out}_metrics.json")


if __name__ == "__main__":
    main()
