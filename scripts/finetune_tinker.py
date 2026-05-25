from __future__ import annotations

import asyncio
import contextlib
import csv
import json
import math
import queue
import random
import re
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import torch
import tinker
from jupyter_client import KernelManager
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    SystemContent,
    ReasoningEffort,
    ToolNamespaceConfig,
    Author,
    Message,
    Role,
    TextContent,
    Conversation,
)
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "openai/gpt-oss-20b"
RENDERER_NAME = "gpt_oss_medium_reasoning"

SYSTEM_PROMPT = """
You are solving hard olympiad-style math problems.

Rules:
1. Work carefully and avoid arithmetic mistakes.
2. Explore, plan, execute, and verify.
3. The final answer must be a non-negative integer from 0 to 99999.
4. Put the final answer inside \\boxed{...}.
5. If you are unsure, still produce your best final boxed answer.

Be concise when possible. Do not ramble needlessly.
""".strip()

TOOL_PROMPT = """
Use the python tool only when it meaningfully helps:
- exact arithmetic
- modular arithmetic
- brute force on small search spaces
- symbolic checks with sympy
- quick verification

Keep code short and use print().
Code should support reasoning, not replace it.
""".strip()

SYSTEM_IDENTITY = "You are GPT-OSS, a careful mathematical reasoning model."

CONTAMINATED_TRAIN_ROWS_AFTER_PASS_FILTER = {35}

def reasoning_effort_from_str(s: str):
    s = (s or "medium").lower().strip()
    if s == "low":
        return ReasoningEffort.LOW
    if s == "medium":
        return ReasoningEffort.MEDIUM
    if s == "high":
        return ReasoningEffort.HIGH
    raise ValueError(f"Unknown reasoning effort: {s}")


def make_harmony_conversation(problem: str, cfg: "DAPOTIRConfig") -> Conversation:
    system_content = SystemContent.new()
    system_content = system_content.with_model_identity(SYSTEM_IDENTITY)

    renderer_name = getattr(cfg, "renderer_name", RENDERER_NAME)
    if "high" in renderer_name:
        reasoning_effort = "high"
    elif "low" in renderer_name:
        reasoning_effort = "low"
    else:
        reasoning_effort = "medium"

    system_content = system_content.with_reasoning_effort(
        reasoning_effort=reasoning_effort_from_str(reasoning_effort)
    )

    if bool(getattr(cfg, "enable_tools", True)):
        tool_config = ToolNamespaceConfig(
            name="python",
            description=TOOL_PROMPT,
            tools=[],
        )
        system_content = system_content.with_tools(tool_config)

    system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
    developer_message = Message.from_role_and_content(Role.DEVELOPER, SYSTEM_PROMPT)
    user_message = Message.from_role_and_content(Role.USER, problem)

    return Conversation.from_messages([system_message, developer_message, user_message])


def harmony_message_text(message: Message) -> str:
    parts: list[str] = []
    for item in getattr(message, "content", []) or []:
        if hasattr(item, "text"):
            parts.append(item.text)
    return "".join(parts)


def make_python_tool_response(output: str, channel: str | None = None) -> Message:
    content = TextContent(text=output)
    author = Author(role=Role.TOOL, name="python")
    msg = Message(author=author, content=[content]).with_recipient("assistant")
    if channel:
        msg = msg.with_channel(channel)
    return msg


def harmony_messages_to_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        try:
            out.append(msg.to_dict())
        except Exception:
            out.append(
                {
                    "role": str(getattr(getattr(msg, "author", None), "role", "")),
                    "recipient": getattr(msg, "recipient", None),
                    "channel": getattr(msg, "channel", None),
                    "content": harmony_message_text(msg),
                }
            )
    return out


def model_input_from_ints(tokens: list[int]) -> tinker.ModelInput:
    return tinker.ModelInput.from_ints(list(tokens))


def harmony_stop_sequences(encoding) -> list[str]:
    stop_strings: list[str] = []
    for token in encoding.stop_tokens_for_assistant_actions():
        try:
            s = encoding.decode_utf8([token])
        except Exception:
            continue
        if s and s not in stop_strings:
            stop_strings.append(s)
    return stop_strings


@dataclass
class DAPOTIRConfig:
    log_path: str = "./tinker_logs/crystal_math_tir_dapo"
    model_name: str = MODEL_NAME
    renderer_name: str = RENDERER_NAME

    # data
    min_medium_pass_rate: float = 0.5
    seed: int = 54
    eval_csv_path: str | None = None

    # training
    learning_rate: float = 1e-6
    warmup_rollout_steps: int = 10
    lora_rank: int = 32  # ok
    max_steps: int = 200
    groups_per_batch: int = 4
    group_size: int = 8  # ok
    trajectory_minibatch_size: int = 128
    ppo_epochs_per_rollout: int = 1
    eps_low: float = 0.2  # ok
    eps_high: float = 0.28  # ok

    # rollout
    max_turns: int = 96  # ok
    context_window: int = 32768  # ok
    context_reserve: int = 128  # ok
    expected_max_length: int = 26_214 # ok
    overlong_cache: int = 6_554  # ok
    python_timeout: float = 8.0  # ok
    max_concurrent_sandboxes: int = 4  # ok (check in smoke, mb crash)

    # sampling
    temperature: float = 1.0  # ok
    top_p: float = 1.0  # ok
    max_dynamic_sampling_groups: int = 64

    max_wall_time_seconds: int | None = 11 * 3600
    final_eval_wall_time_reserve_seconds: int = 1800
    resume_row_skip: int = 0
    resume_state_path: str | None = None
    resume_start_step: int = 0
    resume_with_optimizer: bool = True

    # eval/checkpointing
    print_every: int = 1
    eval_every: int = 10
    milestone_eval_every: int = 50
    checkpoint_every: int = 25
    quick_eval_samples: int = 4
    milestone_eval_samples: int = 16
    final_eval_samples: int = 32
    eval_limit_rows: int | None = 20
    final_eval_limit_rows: int | None = None

    # toggles
    enable_ors: bool = True
    enable_overlong_filtering: bool = False
    enable_tools: bool = True
    enable_shorterbetter: bool = False
    enable_rollout_debug_log: bool = True
    debug_save_text: bool = False

    shorterbetter_alpha: float = 2.0
    shorterbetter_beta: float = 3.5e-5


    @property
    def max_total_response_tokens(self) -> int:
        if self.enable_ors:
            return int(self.expected_max_length + self.overlong_cache)
        return int(self.expected_max_length)


class PythonSandbox:
    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self._km = KernelManager()

        env = os.environ.copy()
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        env["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "0"
        env["JUPYTER_PLATFORM_DIRS"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["MPLBACKEND"] = "Agg"

        self._km.start_kernel(
            env=env,
            extra_arguments=["--Application.log_level=CRITICAL"],
        )

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=max(float(timeout), 20.0))

    def _format_error(self, traceback: list[str]) -> str:
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r"\x1b\[[0-9;]*m", "", frame)
            if 'File "' in clean_frame and "ipython-input" not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return "".join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:
        effective_timeout = float(timeout or self.timeout)
        msg_id = self._client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        start = time.time()
        while True:
            if time.time() - start > effective_timeout:
                with contextlib.suppress(Exception):
                    self._km.interrupt_kernel()
                return f"[ошибка] выполнение превысило лимит по времени: {effective_timeout} секунд"
            try:
                msg = self._client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type = msg.get("msg_type")
            content = msg.get("content", {})
            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                stderr_parts.append(self._format_error(content.get("traceback", [])))
            elif msg_type in {"execute_result", "display_data"}:
                text = content.get("data", {}).get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else text + "\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        if stderr:
            return f"{stdout.rstrip()}\n{stderr}" if stdout else stderr
        return stdout if stdout.strip() else "[предупреждение] нет вывода"

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.stop_channels()
        with contextlib.suppress(Exception):
            self._km.shutdown_kernel(now=True)
        with contextlib.suppress(Exception):
            self._km.cleanup_resources()

    def __del__(self):
        self.close()


def extract_boxed_integer(text: str) -> int | None:
    if not text:
        return None
    matches = re.findall(r"\\boxed\{\s*([0-9,]+)\s*\}", text)
    if not matches:
        return None
    try:
        v = int(matches[-1].replace(",", ""))
    except Exception:
        return None
    return v if 0 <= v <= 99999 else None


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def _get_tool_calls(parsed_message: dict[str, Any]) -> list[Any]:
    tc = parsed_message.get("tool_calls")
    return list(tc) if tc is not None else []


def _tool_call_name(tc: Any) -> str | None:
    if isinstance(tc, dict):
        if "name" in tc:
            return tc["name"]
        if "function" in tc and isinstance(tc["function"], dict):
            return tc["function"].get("name")
    return getattr(tc, "name", None)


def _tool_call_id(tc: Any) -> str | None:
    if isinstance(tc, dict):
        return tc.get("id") or tc.get("tool_call_id")
    return getattr(tc, "id", None) or getattr(tc, "tool_call_id", None)


def _tool_call_to_python_code(tc: Any) -> str:
    raw_args: Any = None
    if isinstance(tc, dict):
        raw_args = tc.get("arguments", tc.get("args"))
        if raw_args is None and isinstance(tc.get("function"), dict):
            raw_args = tc["function"].get("arguments", tc["function"].get("args"))
    else:
        raw_args = getattr(tc, "arguments", None) or getattr(tc, "args", None)
        fn = getattr(tc, "function", None)
        if raw_args is None and fn is not None:
            raw_args = getattr(fn, "arguments", None) or getattr(fn, "args", None)
    if raw_args is None:
        raise ValueError(f"Could not extract tool arguments from {tc!r}")
    if isinstance(raw_args, dict):
        if "code" in raw_args:
            return str(raw_args["code"])
        if "input" in raw_args:
            return str(raw_args["input"])
        if len(raw_args) == 1:
            return str(next(iter(raw_args.values())))
        raise ValueError(f"Tool-call dict has no code field: {raw_args!r}")
    if isinstance(raw_args, str):
        s = raw_args.strip()
        if not s:
            raise ValueError("Empty tool arguments")
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                if "code" in obj:
                    return str(obj["code"])
                if "input" in obj:
                    return str(obj["input"])
                if len(obj) == 1:
                    return str(next(iter(obj.values())))
            if isinstance(obj, str):
                return obj
        except Exception:
            pass
        return s
    return str(raw_args)


def _chunked(xs: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("size must be positive")
    return [xs[i : i + size] for i in range(0, len(xs), size)]


def _extract_last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return _content_to_text(msg.get("content", ""))
    return ""


@dataclass
class AssistantTurnSample:
    prompt: tinker.ModelInput
    tokens: list[int]
    logprobs: list[float]
    text: str
    stop_reason: str | None


@dataclass
class TrajectorySample:
    row: dict[str, Any]
    messages: list[dict[str, Any]]
    assistant_turns: list[AssistantTurnSample]
    predicted_answer: int | None
    is_correct: bool
    base_reward: float
    length_penalty: float
    reward: float
    assistant_tokens: int
    hard_truncated: bool
    python_calls: int
    python_errors: int
    turns: int
    reason: str | None
    max_prompt_tokens: int = 0
    shorterbetter_penalty: float = 0.0
    shorterbetter_sol_tokens: int = 0
    shorterbetter_delta_tokens: int = 0


@dataclass
class ValidGroup:
    row: dict[str, Any]
    trajectories: list[TrajectorySample]
    active_trajectories: list[TrajectorySample]
    n_correct: int
    group_mean_reward: float
    group_std_reward: float
    active_token_count: int


def _normalize_problem_text(question: str) -> str:
    return (question or "").strip()


def load_train_rows(cfg: DAPOTIRConfig) -> list[dict[str, Any]]:
    ds = datasets.load_dataset("ycchen/Crystal-Math-Preview", "rewritten", split="train")
    ds = ds.filter(lambda x: x["medium_pass_rate"] > cfg.min_medium_pass_rate)
    before_decontam = len(ds)
    bad_rows = CONTAMINATED_TRAIN_ROWS_AFTER_PASS_FILTER
    keep_indices = [i for i in range(before_decontam) if i not in bad_rows]
    ds = ds.select(keep_indices)
    rows: list[dict[str, Any]] = []
    for ex in ds:
        rows.append(
            {
                "problem": ex["problem"],
                "answer": int(ex["answer"]),
                "problem_id": "" if ex.get("id") is None else str(ex.get("id")),
                "topic": ex.get("topic"),
            }
        )
    rng = random.Random(cfg.seed)
    rng.shuffle(rows)
    return rows


def apply_shorterbetter_rewards(
    trajectories: list[TrajectorySample],
    cfg: DAPOTIRConfig,
) -> None:
    if not bool(cfg.enable_shorterbetter):
        return

    lengths = [
        int(t.assistant_tokens)
        for t in trajectories
        if int(t.assistant_tokens) > 0
    ]
    if not lengths:
        return

    correct_lengths = [
        int(t.assistant_tokens)
        for t in trajectories
        if t.is_correct and int(t.assistant_tokens) > 0
    ]

    if correct_lengths:
        sol_tokens = int(min(correct_lengths))
    else:
        sol_tokens = int(round(sum(lengths) / max(1, len(lengths))))

    alpha = float(cfg.shorterbetter_alpha)
    beta = float(cfg.shorterbetter_beta)

    for t in trajectories:
        tokens = int(t.assistant_tokens)
        delta = abs(tokens - sol_tokens) if tokens > 0 else 0
        sb_penalty = -beta * float(delta)
        correctness_reward = (alpha / 2.0) if t.is_correct else -(alpha / 2.0)

        t.base_reward = float(correctness_reward)
        t.shorterbetter_penalty = float(sb_penalty)
        t.shorterbetter_sol_tokens = int(sol_tokens)
        t.shorterbetter_delta_tokens = int(delta)
        t.reward = float(t.base_reward) + float(t.length_penalty) + float(t.shorterbetter_penalty)


def load_eval_rows(cfg: DAPOTIRConfig, train_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if cfg.eval_csv_path:
        rows: list[dict[str, Any]] = []
        with open(cfg.eval_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"problem_id", "question", "answer"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"eval csv must contain columns {sorted(required)}")
            for rec in reader:
                rows.append(
                    {
                        "problem": _normalize_problem_text(rec["question"]),
                        "answer": int(str(rec["answer"]).strip()),
                        "problem_id": str(rec["problem_id"]),
                        "topic": None,
                    }
                )
        return rows
    base = list(train_rows or load_train_rows(cfg))
    holdout_n = min(len(base), cfg.eval_limit_rows or 20)
    return base[:holdout_n]


def remove_eval_rows_from_train(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eval_ids = {str(r.get("problem_id", "")) for r in eval_rows if str(r.get("problem_id", ""))}
    eval_questions = {r["problem"].strip() for r in eval_rows}
    out = []
    for row in train_rows:
        pid = str(row.get("problem_id", ""))
        if pid and pid in eval_ids:
            continue
        if row["problem"].strip() in eval_questions:
            continue
        out.append(row)
    return out


class CrystalMathRows:
    def __init__(self, rows_or_cfg: list[dict[str, Any]] | DAPOTIRConfig, seed: int | None = None):
        if isinstance(rows_or_cfg, DAPOTIRConfig):
            rows = load_train_rows(rows_or_cfg)
            seed = rows_or_cfg.seed if seed is None else seed
        else:
            rows = list(rows_or_cfg)
            seed = 42 if seed is None else seed
        self.rows = list(rows)
        self._rng = random.Random(seed)
        self._rng.shuffle(self.rows)
        self._i = 0

    def next_row(self) -> dict[str, Any]:
        if self._i >= len(self.rows):
            self._rng.shuffle(self.rows)
            self._i = 0
        row = self.rows[self._i]
        self._i += 1
        return row


async def rollout_single_trajectory(*, sampling_client: tinker.SamplingClient, encoding, row: dict[str, Any], cfg: DAPOTIRConfig) -> TrajectorySample:
    conversation = make_harmony_conversation(row["problem"], cfg)
    assistant_turns: list[AssistantTurnSample] = []

    use_tools = bool(getattr(cfg, "enable_tools", True))
    sandbox = PythonSandbox(timeout=cfg.python_timeout) if use_tools else None

    assistant_tokens_total = 0
    python_calls = 0
    python_errors = 0
    turns = 0
    hard_truncated = False
    reason: str | None = None
    max_prompt_tokens_seen = 0
    final_text_parts: list[str] = []
    stop_sequences = harmony_stop_sequences(encoding)

    try:
        while True:
            if turns >= cfg.max_turns:
                hard_truncated = True
                reason = "max_turns"
                break
            if assistant_tokens_total >= cfg.max_total_response_tokens:
                hard_truncated = True
                reason = "max_total_response_tokens"
                break

            prompt_ids = list(encoding.render_conversation_for_completion(conversation, Role.ASSISTANT))
            prompt = model_input_from_ints(prompt_ids)

            max_prompt_tokens_seen = max(max_prompt_tokens_seen, int(prompt.length))

            response_available = int(cfg.max_total_response_tokens) - int(assistant_tokens_total)
            context_available = int(cfg.context_window) - int(prompt.length) - int(cfg.context_reserve)

            if response_available <= 0:
                hard_truncated = True
                reason = "max_total_response_tokens"
                break

            if context_available <= 0:
                hard_truncated = True
                reason = "context_window"
                break

            per_turn_budget = int(min(response_available, context_available))

            sampling_params = tinker.SamplingParams(
                max_tokens=per_turn_budget,
                stop=stop_sequences,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
            result = await sampling_client.sample_async(prompt=prompt, num_samples=1, sampling_params=sampling_params)
            seq = result.sequences[0]
            turns += 1
            tokens = list(seq.tokens)
            logprobs = list(getattr(seq, "logprobs", []) or [])
            if len(logprobs) != len(tokens):
                raise RuntimeError(
                    f"Tinker returned {len(logprobs)} logprobs for {len(tokens)} tokens"
                )

            stop_reason = getattr(seq, "stop_reason", None)
            assistant_tokens_total += len(tokens)
            if assistant_tokens_total >= cfg.max_total_response_tokens and stop_reason == "length":
                hard_truncated = True

            try:
                assistant_text = encoding.decode_utf8(tokens)
            except Exception as exc:
                assistant_text = f"[DECODE_ERROR] {type(exc).__name__}: {exc}"
            assistant_turns.append(
                AssistantTurnSample(
                    prompt=prompt,
                    tokens=tokens,
                    logprobs=logprobs,
                    text=assistant_text,
                    stop_reason=stop_reason,
                )
            )

            try:
                new_messages = encoding.parse_messages_from_completion_tokens(
                    tokens,
                    Role.ASSISTANT,
                    strict=False,
                )
            except TypeError:
                new_messages = encoding.parse_messages_from_completion_tokens(tokens, Role.ASSISTANT)
            except Exception:
                final_text_parts.append(assistant_text)
                hard_truncated = True
                reason = "parse_failed"
                break

            if not new_messages:
                final_text_parts.append(assistant_text)
                hard_truncated = True
                reason = "parse_failed"
                break

            conversation.messages.extend(new_messages)
            last_message = new_messages[-1]
            last_text = harmony_message_text(last_message)
            final_text_parts.append(last_text)

            if getattr(last_message, "channel", None) == "final":
                reason = "final_answer"
                break

            if getattr(last_message, "recipient", None) == "python":
                if not use_tools:
                    hard_truncated = True
                    reason = "tool_call_disabled"
                    break

                python_calls += 1
                try:
                    assert sandbox is not None
                    output = sandbox.execute(last_text, timeout=cfg.python_timeout)
                except Exception as exc:
                    output = f"[ERROR] {exc}"
                if output.startswith("[ERROR]") or "Traceback" in output:
                    python_errors += 1

                tool_response = make_python_tool_response(
                    output,
                    channel=getattr(last_message, "channel", None),
                )
                conversation.messages.append(tool_response)
                continue

            if stop_reason == "length":
                hard_truncated = True
                reason = "truncated_generation"
                break

            continue

        final_text = "\n".join(final_text_parts)
        pred = extract_boxed_integer(final_text)
        is_correct = pred is not None and pred == int(row["answer"])
        base_reward = 1.0 if is_correct else -1.0
        if not cfg.enable_ors:
            length_penalty = 0.0
        elif assistant_tokens_total <= cfg.expected_max_length:
            length_penalty = 0.0
        elif assistant_tokens_total <= cfg.max_total_response_tokens:
            length_penalty = float(cfg.expected_max_length - assistant_tokens_total) / float(max(1, cfg.overlong_cache))
        else:
            length_penalty = -1.0
        reward = base_reward + length_penalty
        return TrajectorySample(
            row=row,
            messages=harmony_messages_to_dicts(conversation.messages),
            assistant_turns=assistant_turns,
            predicted_answer=pred,
            is_correct=is_correct,
            base_reward=base_reward,
            length_penalty=length_penalty,
            reward=reward,
            assistant_tokens=assistant_tokens_total,
            hard_truncated=hard_truncated,
            python_calls=python_calls,
            python_errors=python_errors,
            turns=turns,
            reason=reason,
            max_prompt_tokens=max_prompt_tokens_seen,
        )
    finally:
        if sandbox is not None:
            sandbox.close()


async def rollout_group(
    *,
    sampling_client: tinker.SamplingClient,
    encoding,
    row: dict[str, Any],
    cfg: DAPOTIRConfig,
    semaphore: asyncio.Semaphore | None = None,
) -> ValidGroup | None:
    sem = semaphore or asyncio.Semaphore(max(1, int(cfg.max_concurrent_sandboxes)))

    async def _run_one() -> TrajectorySample:
        async with sem:
            return await rollout_single_trajectory(
                sampling_client=sampling_client,
                encoding=encoding,
                row=row,
                cfg=cfg,
            )

    trajectories = list(await asyncio.gather(*[_run_one() for _ in range(cfg.group_size)]))
    n_correct = sum(int(t.is_correct) for t in trajectories)
    if not (0 < n_correct < cfg.group_size):
        return None

    if bool(cfg.enable_shorterbetter):
        apply_shorterbetter_rewards(trajectories, cfg)

    if cfg.enable_overlong_filtering:
        active = [t for t in trajectories if (not t.hard_truncated) and t.assistant_tokens > 0]
    else:
        active = [t for t in trajectories if t.assistant_tokens > 0]
    if not active:
        return None
    rewards = [t.reward for t in active]
    mean_reward = float(sum(rewards) / len(rewards))
    variance = float(sum((r - mean_reward) ** 2 for r in rewards) / len(rewards))
    std_reward = math.sqrt(variance)
    if std_reward <= 1e-8:
        return None
    active_token_count = sum(t.assistant_tokens for t in active)
    if active_token_count <= 0:
        return None
    return ValidGroup(
        row=row,
        trajectories=trajectories,
        active_trajectories=active,
        n_correct=n_correct,
        group_mean_reward=mean_reward,
        group_std_reward=std_reward,
        active_token_count=active_token_count,
    )


def build_turn_datum(*, turn: AssistantTurnSample, scaled_advantage: float) -> tinker.Datum:
    prompt_len_minus_one = turn.prompt.length - 1

    model_input = turn.prompt.append(
        tinker.EncodedTextChunk(tokens=turn.tokens[:-1])
    )

    target_tokens = [0] * prompt_len_minus_one + list(turn.tokens)
    logprobs = [0.0] * prompt_len_minus_one + list(turn.logprobs)
    advantages = [0.0] * prompt_len_minus_one + [float(scaled_advantage)] * len(turn.tokens)

    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "logprobs": logprobs,
            "advantages": advantages,
        },
    )


def build_training_datums(groups: list[ValidGroup]) -> tuple[list[tinker.Datum], dict[str, float]]:
    datums: list[tinker.Datum] = []

    batch_active_token_count = sum(g.active_token_count for g in groups)
    if batch_active_token_count <= 0:
        return [], {}
    
    total_active_trajectories = 0
    total_active_tokens = 0
    total_correct = 0
    total_rewards = []
    total_base_rewards = []
    total_length_penalties = []
    total_shorterbetter_penalties = []
    total_shorterbetter_delta_tokens = []
    total_shorterbetter_sol_tokens = []
    total_truncated = 0
    total_python_calls = 0
    total_python_errors = 0
    sampled_logprobs = []
    for group in groups:
        total_active_trajectories += len(group.active_trajectories)
        total_active_tokens += group.active_token_count
        total_correct += sum(int(t.is_correct) for t in group.trajectories)
        total_truncated += sum(int(t.hard_truncated) for t in group.trajectories)
        total_python_calls += sum(t.python_calls for t in group.trajectories)
        total_python_errors += sum(t.python_errors for t in group.trajectories)
        total_rewards.extend(t.reward for t in group.trajectories)
        total_base_rewards.extend(t.base_reward for t in group.trajectories)
        total_length_penalties.extend(t.length_penalty for t in group.trajectories)
        total_shorterbetter_penalties.extend(
            getattr(t, "shorterbetter_penalty", 0.0)
            for t in group.trajectories
        )
        total_shorterbetter_delta_tokens.extend(
            getattr(t, "shorterbetter_delta_tokens", 0)
            for t in group.trajectories
        )
        total_shorterbetter_sol_tokens.extend(
            getattr(t, "shorterbetter_sol_tokens", 0)
            for t in group.trajectories
            if getattr(t, "shorterbetter_sol_tokens", 0) > 0
        )
        for traj in group.active_trajectories:
            standardized_advantage = (traj.reward - group.group_mean_reward) / group.group_std_reward
            scaled_advantage = standardized_advantage / float(batch_active_token_count)
            for turn in traj.assistant_turns:
                datums.append(build_turn_datum(turn=turn, scaled_advantage=scaled_advantage))
                if turn.logprobs:
                    sampled_logprobs.extend(turn.logprobs)
    metrics = {
        "rollout/valid_groups": float(len(groups)),
        "rollout/active_trajectories": float(total_active_trajectories),
        "rollout/active_tokens": float(total_active_tokens),
        "rollout/group_acc_mean": float(total_correct) / float(sum(len(g.trajectories) for g in groups)) if groups else 0.0,
        "rollout/reward_mean": float(sum(total_rewards) / len(total_rewards)) if total_rewards else 0.0,
        "rollout/base_reward_mean": float(sum(total_base_rewards) / len(total_base_rewards)) if total_base_rewards else 0.0,
        "rollout/length_penalty_mean": float(sum(total_length_penalties) / len(total_length_penalties)) if total_length_penalties else 0.0,
        "rollout/shorterbetter_penalty_mean": float(sum(total_shorterbetter_penalties) / len(total_shorterbetter_penalties)) if total_shorterbetter_penalties else 0.0,
        "rollout/shorterbetter_delta_tokens_mean": float(sum(total_shorterbetter_delta_tokens) / len(total_shorterbetter_delta_tokens)) if total_shorterbetter_delta_tokens else 0.0,
        "rollout/shorterbetter_sol_tokens_mean": float(sum(total_shorterbetter_sol_tokens) / len(total_shorterbetter_sol_tokens)) if total_shorterbetter_sol_tokens else 0.0,
        "rollout/hard_trunc_frac": float(total_truncated) / float(sum(len(g.trajectories) for g in groups)) if groups else 0.0,
        "rollout/python_calls": float(total_python_calls),
        "rollout/python_errors": float(total_python_errors),
        "rollout/mean_turns": float(sum(t.turns for g in groups for t in g.trajectories)) / float(sum(len(g.trajectories) for g in groups)) if groups else 0.0,
        "train/datums": float(len(datums)),
        "rollout/neg_mean_sampled_logprob": -float(sum(sampled_logprobs) / len(sampled_logprobs)) if sampled_logprobs else 0.0,
        "rollout/max_prompt_tokens": float(
            max((t.max_prompt_tokens for g in groups for t in g.trajectories), default=0)
        ),
        "rollout/context_hit_frac": float(
            sum(int(t.reason == "context_window") for g in groups for t in g.trajectories)
        ) / float(sum(len(g.trajectories) for g in groups)) if groups else 0.0,
        "rollout/response_cap_hit_frac": float(
            sum(int(t.reason == "max_total_response_tokens") for g in groups for t in g.trajectories)
        ) / float(sum(len(g.trajectories) for g in groups)) if groups else 0.0,
        "rollout/python_calls_per_traj": float(total_python_calls) / float(sum(len(g.trajectories) for g in groups)) if groups else 0.0,
        "rollout/python_errors_per_traj": float(total_python_errors) / float(sum(len(g.trajectories) for g in groups)) if groups else 0.0,
        "rollout/python_error_rate": float(total_python_errors) / float(max(1, total_python_calls)),
    }
    return datums, metrics


def get_ppo_loss_config(cfg: DAPOTIRConfig) -> dict[str, float]:
    return {
        "clip_low_threshold": 1.0 - float(cfg.eps_low),
        "clip_high_threshold": 1.0 + float(cfg.eps_high),
    }


async def collect_valid_groups(
    *,
    rows: CrystalMathRows,
    sampling_client: tinker.SamplingClient,
    encoding,
    cfg: DAPOTIRConfig,
) -> tuple[list[ValidGroup], dict[str, float]]:
    valid_groups: list[ValidGroup] = []
    sampled_groups = 0
    rejected_groups = 0

    global_semaphore = asyncio.Semaphore(max(1, int(cfg.max_concurrent_sandboxes)))
    max_groups = max(int(cfg.groups_per_batch), int(cfg.max_dynamic_sampling_groups))

    while len(valid_groups) < cfg.groups_per_batch and sampled_groups < max_groups:
        needed_valid = cfg.groups_per_batch - len(valid_groups)
        remaining_budget = max_groups - sampled_groups
        n_to_sample = min(needed_valid, remaining_budget)

        batch_rows = [rows.next_row() for _ in range(n_to_sample)]

        results = await asyncio.gather(*[
            rollout_group(
                sampling_client=sampling_client,
                encoding=encoding,
                row=row,
                cfg=cfg,
                semaphore=global_semaphore,
            )
            for row in batch_rows
        ])

        sampled_groups += len(results)

        for group in results:
            if group is None:
                rejected_groups += 1
            else:
                valid_groups.append(group)

    accept_rate = float(len(valid_groups)) / float(sampled_groups) if sampled_groups else 0.0
    exhausted = len(valid_groups) < cfg.groups_per_batch

    metrics = {
        "rollout/sampled_groups": float(sampled_groups),
        "rollout/rejected_groups": float(rejected_groups),
        "rollout/accept_rate": accept_rate,
        "rollout/dynamic_sampling_exhausted": float(exhausted),
    }

    if exhausted:
        raise RuntimeError(
            f"Dynamic sampling exhausted: got {len(valid_groups)} valid groups "
            f"out of required {cfg.groups_per_batch} after sampling {sampled_groups} groups. "
            f"accept_rate={accept_rate:.4f}. "
            f"Try increasing max_dynamic_sampling_groups or raising min_medium_pass_rate."
        )

    return valid_groups, metrics


def _make_adam_params(cfg: DAPOTIRConfig, step: int) -> tinker.AdamParams:
    warm = max(1, int(cfg.warmup_rollout_steps))
    warm_frac = min(1.0, float(step + 1) / float(warm))
    lr = cfg.learning_rate * warm_frac
    return tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)


async def eval_avgk(*, sampling_client: tinker.SamplingClient, encoding, rows: list[dict[str, Any]], cfg: DAPOTIRConfig, k: int = 4, limit: int | None = None) -> dict[str, float]:
    use_rows = rows[:limit] if limit is not None else rows
    grouped: list[list[TrajectorySample]] = []
    for row in use_rows:
        trajs = []
        for _ in range(k):
            trajs.append(await rollout_single_trajectory(sampling_client=sampling_client, encoding=encoding, row=row, cfg=cfg))
        grouped.append(trajs)
    flat = [t for group in grouped for t in group]
    avg_k = sum(float(t.is_correct) for t in flat) / max(1, len(flat))
    pass_k = sum(float(any(t.is_correct for t in group)) for group in grouped) / max(1, len(grouped))
    return {
        f"eval/avg@{k}": avg_k,
        f"eval/pass@{k}": pass_k,
        "eval/mean_tokens": sum(t.assistant_tokens for t in flat) / max(1, len(flat)),
        "eval/mean_turns": sum(t.turns for t in flat) / max(1, len(flat)),
        "eval/hard_trunc_frac": sum(float(t.hard_truncated) for t in flat) / max(1, len(flat)),
        "eval/tool_error_rate": sum(t.python_errors for t in flat) / max(1, sum(t.python_calls for t in flat)),
        "eval/n_rows": float(len(use_rows)),
        "eval/max_prompt_tokens": max((t.max_prompt_tokens for t in flat), default=0),
        "eval/context_hit_frac": sum(float(t.reason == "context_window") for t in flat) / max(1, len(flat)),
        "eval/response_cap_hit_frac": sum(float(t.reason == "max_total_response_tokens") for t in flat) / max(1, len(flat)),
    }


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _role_to_str(x: Any) -> str:
    try:
        return str(x).lower()
    except Exception:
        return ""


def _message_content_chars(msg: dict[str, Any]) -> int:
    try:
        return len(_content_to_text(msg.get("content", "")))
    except Exception:
        return 0


def _is_tool_message(msg: dict[str, Any]) -> bool:
    role_s = _role_to_str(msg.get("role", ""))
    author_s = _role_to_str(msg.get("author", ""))
    name_s = _role_to_str(msg.get("name", ""))
    return ("tool" in role_s) or ("tool" in author_s) or (name_s == "python")


def _write_rollout_debug(
    path: Path,
    *,
    step: int,
    groups: list[ValidGroup],
    cfg: DAPOTIRConfig,
) -> None:
    if not bool(getattr(cfg, "enable_rollout_debug_log", True)):
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    batch_active_token_count = sum(g.active_token_count for g in groups)
    save_text = bool(getattr(cfg, "debug_save_text", False))

    with path.open("a", encoding="utf-8") as f:
        for group_idx, group in enumerate(groups):
            trajs = list(group.trajectories)
            active_ids = {id(t) for t in group.active_trajectories}

            correct_trajs = [t for t in trajs if t.is_correct]
            wrong_trajs = [t for t in trajs if not t.is_correct]

            def _mean(xs):
                return float(sum(xs) / len(xs)) if xs else 0.0

            group_rec = {
                "record_type": "group",
                "step": int(step),
                "group_idx": int(group_idx),
                "problem_id": str(group.row.get("problem_id", "")),
                "topic": group.row.get("topic"),
                "gold_answer": int(group.row.get("answer", -1)),

                "n_trajectories": int(len(trajs)),
                "n_active_trajectories": int(len(group.active_trajectories)),
                "n_correct": int(group.n_correct),
                "group_acc": float(group.n_correct / max(1, len(trajs))),

                "group_mean_reward": float(group.group_mean_reward),
                "group_std_reward": float(group.group_std_reward),
                "active_token_count": int(group.active_token_count),
                "batch_active_token_count": int(batch_active_token_count),

                "sol_tokens": int(max(
                    [getattr(t, "shorterbetter_sol_tokens", 0) for t in trajs],
                    default=0,
                )),
                "mean_tokens": _mean([t.assistant_tokens for t in trajs]),
                "mean_tokens_correct": _mean([t.assistant_tokens for t in correct_trajs]),
                "mean_tokens_wrong": _mean([t.assistant_tokens for t in wrong_trajs]),
                "min_tokens_correct": int(min([t.assistant_tokens for t in correct_trajs], default=0)),
                "min_tokens_wrong": int(min([t.assistant_tokens for t in wrong_trajs], default=0)),

                "mean_reward": _mean([t.reward for t in trajs]),
                "mean_base_reward": _mean([t.base_reward for t in trajs]),
                "mean_ors_penalty": _mean([t.length_penalty for t in trajs]),
                "mean_sb_penalty": _mean([getattr(t, "shorterbetter_penalty", 0.0) for t in trajs]),
                "mean_sb_delta_tokens": _mean([getattr(t, "shorterbetter_delta_tokens", 0) for t in trajs]),

                "python_calls": int(sum(t.python_calls for t in trajs)),
                "python_errors": int(sum(t.python_errors for t in trajs)),
                "python_error_rate": float(
                    sum(t.python_errors for t in trajs) / max(1, sum(t.python_calls for t in trajs))
                ),
                "mean_turns": _mean([t.turns for t in trajs]),
                "hard_trunc_frac": _mean([float(t.hard_truncated) for t in trajs]),
                "context_hit_frac": _mean([float(t.reason == "context_window") for t in trajs]),
                "response_cap_hit_frac": _mean([float(t.reason == "max_total_response_tokens") for t in trajs]),

                "enable_ors": bool(cfg.enable_ors),
                "enable_tools": bool(getattr(cfg, "enable_tools", True)),
                "enable_shorterbetter": bool(cfg.enable_shorterbetter),
                "shorterbetter_alpha": float(cfg.shorterbetter_alpha),
                "shorterbetter_beta": float(cfg.shorterbetter_beta),
            }
            f.write(json.dumps(group_rec, ensure_ascii=False) + "\n")

            for traj_idx, t in enumerate(trajs):
                standardized_advantage = 0.0
                if group.group_std_reward > 1e-8:
                    standardized_advantage = float((t.reward - group.group_mean_reward) / group.group_std_reward)

                scaled_advantage = 0.0
                if batch_active_token_count > 0:
                    scaled_advantage = float(standardized_advantage / float(batch_active_token_count))

                tool_output_chars = int(sum(
                    _message_content_chars(m)
                    for m in (t.messages or [])
                    if isinstance(m, dict) and _is_tool_message(m)
                ))

                assistant_text_chars = int(sum(len(turn.text or "") for turn in t.assistant_turns))
                first_stop_reason = t.assistant_turns[0].stop_reason if t.assistant_turns else None
                last_stop_reason = t.assistant_turns[-1].stop_reason if t.assistant_turns else None

                rec = {
                    "record_type": "trajectory",
                    "step": int(step),
                    "group_idx": int(group_idx),
                    "traj_idx": int(traj_idx),
                    "problem_id": str(group.row.get("problem_id", "")),
                    "topic": group.row.get("topic"),
                    "gold_answer": int(group.row.get("answer", -1)),

                    "is_active": int(id(t) in active_ids),
                    "is_correct": int(t.is_correct),
                    "predicted_answer": t.predicted_answer,

                    "reward": float(t.reward),
                    "base_reward": float(t.base_reward),
                    "ors_length_penalty": float(t.length_penalty),
                    "shorterbetter_penalty": float(getattr(t, "shorterbetter_penalty", 0.0)),
                    "shorterbetter_sol_tokens": int(getattr(t, "shorterbetter_sol_tokens", 0)),
                    "shorterbetter_delta_tokens": int(getattr(t, "shorterbetter_delta_tokens", 0)),

                    "group_mean_reward": float(group.group_mean_reward),
                    "group_std_reward": float(group.group_std_reward),
                    "standardized_advantage": float(standardized_advantage),
                    "scaled_advantage": float(scaled_advantage),

                    "assistant_tokens": int(t.assistant_tokens),
                    "assistant_text_chars": int(assistant_text_chars),
                    "tool_output_chars": int(tool_output_chars),
                    "turns": int(t.turns),
                    "python_calls": int(t.python_calls),
                    "python_errors": int(t.python_errors),
                    "python_error_rate": float(t.python_errors / max(1, t.python_calls)),

                    "hard_truncated": int(t.hard_truncated),
                    "reason": t.reason,
                    "max_prompt_tokens": int(t.max_prompt_tokens),
                    "first_stop_reason": first_stop_reason,
                    "last_stop_reason": last_stop_reason,

                    "enable_ors": bool(cfg.enable_ors),
                    "enable_tools": bool(getattr(cfg, "enable_tools", True)),
                    "enable_shorterbetter": bool(cfg.enable_shorterbetter),
                    "shorterbetter_alpha": float(cfg.shorterbetter_alpha),
                    "shorterbetter_beta": float(cfg.shorterbetter_beta),
                }

                if save_text:
                    rec["assistant_text"] = "\n".join(turn.text or "" for turn in t.assistant_turns)

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def run_training(cfg: DAPOTIRConfig) -> None:
    log_dir = Path(cfg.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    train_rows = load_train_rows(cfg)
    eval_rows = load_eval_rows(cfg, train_rows)
    train_rows = remove_eval_rows_from_train(train_rows, eval_rows)
    row_stream = CrystalMathRows(train_rows, seed=cfg.seed)

    if int(cfg.resume_row_skip) > 0:
        n_skip = int(cfg.resume_row_skip)
        for _ in range(n_skip):
            row_stream.next_row()
        print(f"пропущено {n_skip} строк перед рестартом")

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    service_client = tinker.ServiceClient()

    if cfg.resume_state_path:
        if cfg.resume_with_optimizer:
            training_client = await service_client.create_training_client_from_state_with_optimizer_async(
                cfg.resume_state_path
            )
        else:
            training_client = await service_client.create_training_client_from_state_async(
                cfg.resume_state_path
            )
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=cfg.model_name,
            rank=cfg.lora_rank,
        )
    train_start_time = time.time()
    last_sampling_client = None
    ppo_loss_config = get_ppo_loss_config(cfg)

    for step in range(cfg.resume_start_step + 1, cfg.max_steps + 1):
        if cfg.max_wall_time_seconds is not None:
            elapsed = time.time() - train_start_time
            if elapsed >= cfg.max_wall_time_seconds:
                print(f"ранняя остановка из-за лимита по времени: {elapsed/3600:.2f}ч")
                break
        step_metrics: dict[str, float] = {"progress/step": float(step)}
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(name=f"step-{step}")
        last_sampling_client = sampling_client
        valid_groups, dyn_metrics = await collect_valid_groups(rows=row_stream, sampling_client=sampling_client, encoding=encoding, cfg=cfg)
        step_metrics.update(dyn_metrics)
        datums, batch_metrics = build_training_datums(valid_groups)
        step_metrics.update(batch_metrics)
        _write_rollout_debug(
            log_dir / "rollout_debug.jsonl",
            step=step,
            groups=valid_groups,
            cfg=cfg,
        )
        if datums:
            minibatches = _chunked(datums, max(1, cfg.trajectory_minibatch_size))
            for epoch in range(cfg.ppo_epochs_per_rollout):
                random.Random(cfg.seed + step + epoch).shuffle(minibatches)
                for mb in minibatches:
                    adam_params = _make_adam_params(cfg, step - 1)
                    fwd_future = await training_client.forward_backward_async(mb, loss_fn="ppo", loss_fn_config=ppo_loss_config)
                    opt_future = await training_client.optim_step_async(adam_params)
                    fwd_res = await fwd_future.result_async()
                    opt_res = await opt_future.result_async()
                    if getattr(fwd_res, "metrics", None):
                        for k, v in fwd_res.metrics.items():
                            with contextlib.suppress(Exception):
                                step_metrics[f"train/{k}"] = float(v)
                    if getattr(opt_res, "metrics", None):
                        for k, v in opt_res.metrics.items():
                            with contextlib.suppress(Exception):
                                step_metrics[f"optim/{k}"] = float(v)
                    step_metrics["optim/lr"] = float(adam_params.learning_rate)

        def has_time_for_eval() -> bool:
            if cfg.max_wall_time_seconds is None:
                return True
            elapsed = time.time() - train_start_time
            return elapsed < cfg.max_wall_time_seconds - cfg.final_eval_wall_time_reserve_seconds
        
        if step % cfg.eval_every == 0 and has_time_for_eval():
            step_metrics.update({f"quick_{k}": v for k, v in (await eval_avgk(sampling_client=sampling_client, encoding=encoding, rows=eval_rows, cfg=cfg, k=cfg.quick_eval_samples, limit=cfg.eval_limit_rows)).items()})
        if step % cfg.milestone_eval_every == 0 and has_time_for_eval():
            step_metrics.update({f"milestone_{k}": v for k, v in (await eval_avgk(sampling_client=sampling_client, encoding=encoding, rows=eval_rows, cfg=cfg, k=cfg.milestone_eval_samples, limit=cfg.eval_limit_rows)).items()})
        if step % cfg.checkpoint_every == 0:
            save_res = await (await training_client.save_state_async(name=f"checkpoint-{step:05d}")).result_async()
            ckpt_record = {"step": step, "path": getattr(save_res, "path", "")}
            _write_jsonl(log_dir / "checkpoints.jsonl", ckpt_record)
        _write_jsonl(log_dir / "metrics.jsonl", step_metrics)
        if step % max(1, cfg.print_every) == 0:
            active_traj = max(1.0, step_metrics.get("rollout/active_trajectories", 1.0))
            tok_per_traj = step_metrics.get("rollout/active_tokens", 0.0) / active_traj
            py_per_traj = step_metrics.get("rollout/python_calls", 0.0) / active_traj

            print(" | ".join([
                f"step={step}",
                f"accept={step_metrics.get('rollout/accept_rate', 0.0):.2f}",
                f"gacc={step_metrics.get('rollout/group_acc_mean', 0.0):.2f}",
                f"reward={step_metrics.get('rollout/reward_mean', 0.0):.3f}",
                f"lp={step_metrics.get('rollout/length_penalty_mean', 0.0):.3f}",
                f"sb={step_metrics.get('rollout/shorterbetter_penalty_mean', 0.0):.3f}",
                f"sbd={step_metrics.get('rollout/shorterbetter_delta_tokens_mean', 0.0):.0f}",
                f"trunc={step_metrics.get('rollout/hard_trunc_frac', 0.0):.1%}",
                f"ctx={step_metrics.get('rollout/context_hit_frac', 0.0):.1%}",
                f"cap={step_metrics.get('rollout/response_cap_hit_frac', 0.0):.1%}",
                f"turns={step_metrics.get('rollout/mean_turns', 0.0):.1f}",
                f"py={step_metrics.get('rollout/python_calls', 0.0):.0f}",
                f"py/traj={py_per_traj:.2f}",
                f"pyerr={step_metrics.get('rollout/python_errors', 0.0):.0f}",
                f"tok/traj={tok_per_traj:.0f}",
                f"maxp={int(step_metrics.get('rollout/max_prompt_tokens', 0.0))}",
                f"datums={int(step_metrics.get('train/datums', 0.0))}",
                f"lr={step_metrics.get('optim/lr', 0.0):.2e}",
            ]))

    if cfg.max_wall_time_seconds is not None:
        elapsed = time.time() - train_start_time
        if elapsed > cfg.max_wall_time_seconds - cfg.final_eval_wall_time_reserve_seconds:
            print("пропускаю финальную оценку из-за лимита по времени")
            return

    final_sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="final")
    final_limit = cfg.final_eval_limit_rows if cfg.final_eval_limit_rows is not None else cfg.eval_limit_rows
    final_metrics = await eval_avgk(
        sampling_client=final_sampling_client,
        encoding=encoding,
        rows=eval_rows,
        cfg=cfg,
        k=cfg.final_eval_samples,
        limit=final_limit,
    )
    _write_jsonl(log_dir / "metrics.jsonl", {f"final_{k}": v for k, v in final_metrics.items()})


def make_config(**kwargs: Any) -> DAPOTIRConfig:
    return DAPOTIRConfig(**kwargs)


async def run_default() -> None:
    await run_training(make_config())


if __name__ == "__main__":
    asyncio.run(run_default())
