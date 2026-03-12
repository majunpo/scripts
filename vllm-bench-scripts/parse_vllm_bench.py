#!/usr/bin/env python3
"""
VLLM Benchmark Log Parser
=========================
解析 vllm bench serve 产生的日志文件，提取关键性能指标并输出到 CSV。

支持功能：
  - 单个或批量 log 文件解析
  - 自动从文件名/日志头部提取模型名、input_len、output_len
  - 输出到 CSV（默认与 log 同目录）或指定路径
  - 追加模式：多次运行可合并结果到同一 CSV

用法:
  python parse_vllm_bench.py <log_file_or_dir> [options]

示例:
  # 解析单个文件
  python parse_vllm_bench.py vllm_bench_xxx.log

  # 解析目录下所有 .log 文件
  python parse_vllm_bench.py ./logs/

  # 指定输出文件
  python parse_vllm_bench.py vllm_bench_xxx.log -o results.csv

  # 追加模式
  python parse_vllm_bench.py vllm_bench_xxx.log -o results.csv --append
"""

import re
import csv
import sys
import argparse
from pathlib import Path
from typing import Optional


# CSV 列定义（有序）
CSV_FIELDS = [
    "log_file",
    "model",
    "random_input_len",
    "random_output_len",
    "random_range_ratio",
    "test_id",
    "num_prompts",
    "max_concurrency",
    "successful_requests",
    "failed_requests",
    "benchmark_duration_s",
    "total_input_tokens",
    "total_generated_tokens",
    "request_throughput_req_s",
    "output_token_throughput_tok_s",
    "peak_output_token_throughput_tok_s",
    "peak_concurrent_requests",
    "total_token_throughput_tok_s",
    "ttft_mean_ms",
    "ttft_median_ms",
    "ttft_p99_ms",
    "tpot_mean_ms",
    "tpot_median_ms",
    "tpot_p99_ms",
    "itl_mean_ms",
    "itl_median_ms",
    "itl_p99_ms",
    "test_start_time",
    "test_end_time",
]


def extract_header_info(content: str) -> dict:
    """从日志头部提取全局配置信息。"""
    info = {}

    m = re.search(r"MODEL:\s*(.+)", content)
    info["model"] = m.group(1).strip() if m else ""

    m = re.search(r"RANDOM_INPUT_LEN:\s*(\d+)", content)
    info["random_input_len"] = m.group(1) if m else ""

    m = re.search(r"RANDOM_OUTPUT_LEN:\s*(\d+)", content)
    info["random_output_len"] = m.group(1) if m else ""

    m = re.search(r"RANDOM_RANGE_RATIO:\s*([\d.]+)", content)
    info["random_range_ratio"] = m.group(1) if m else ""

    return info


def extract_test_blocks(content: str) -> list[dict]:
    """
    将日志按 TEST [n/N] START ... TEST [n/N] END 分割成独立的测试块，
    然后从每个块中提取指标。
    """
    results = []

    # 匹配每个测试块（从 TEST START 到 TEST END）
    block_pattern = (
        r"TEST \[(\d+)/\d+\] START:\s*(.+?)\n"
        r"Configuration:\s*num-prompts=(\d+),\s*max-concurrency=(\d+)"
        r"(.*?)"
        r"TEST \[\1/\d+\] END:\s*(.+?)\n"
    )

    for m in re.finditer(block_pattern, content, re.DOTALL):
        test_id = m.group(1)
        start_time = m.group(2).strip()
        num_prompts = m.group(3)
        max_concurrency = m.group(4)
        block_body = m.group(5)
        end_time = m.group(6).strip()

        metrics = _extract_metrics(block_body)
        if metrics is None:
            # 没有找到 benchmark result 部分（可能测试失败），跳过
            continue

        record = {
            "test_id": test_id,
            "num_prompts": num_prompts,
            "max_concurrency": max_concurrency,
            "test_start_time": start_time,
            "test_end_time": end_time,
        }
        record.update(metrics)
        results.append(record)

    return results


def _extract_metrics(block: str) -> Optional[dict]:
    """从单个测试块的 benchmark result 区域提取所有指标。"""

    # 定义要提取的指标：(csv_field_name, regex_pattern)
    metric_patterns = [
        ("successful_requests",                r"Successful requests:\s*(\d+)"),
        ("failed_requests",                    r"Failed requests:\s*(\d+)"),
        ("benchmark_duration_s",               r"Benchmark duration \(s\):\s*([\d.]+)"),
        ("total_input_tokens",                 r"Total input tokens:\s*(\d+)"),
        ("total_generated_tokens",             r"Total generated tokens:\s*(\d+)"),
        ("request_throughput_req_s",           r"Request throughput \(req/s\):\s*([\d.]+)"),
        ("output_token_throughput_tok_s",      r"Output token throughput \(tok/s\):\s*([\d.]+)"),
        ("peak_output_token_throughput_tok_s", r"Peak output token throughput \(tok/s\):\s*([\d.]+)"),
        ("peak_concurrent_requests",           r"Peak concurrent requests:\s*([\d.]+)"),
        ("total_token_throughput_tok_s",       r"Total token throughput \(tok/s\):\s*([\d.]+)"),
        ("ttft_mean_ms",                       r"Mean TTFT \(ms\):\s*([\d.]+)"),
        ("ttft_median_ms",                     r"Median TTFT \(ms\):\s*([\d.]+)"),
        ("ttft_p99_ms",                        r"P99 TTFT \(ms\):\s*([\d.]+)"),
        ("tpot_mean_ms",                       r"Mean TPOT \(ms\):\s*([\d.]+)"),
        ("tpot_median_ms",                     r"Median TPOT \(ms\):\s*([\d.]+)"),
        ("tpot_p99_ms",                        r"P99 TPOT \(ms\):\s*([\d.]+)"),
        ("itl_mean_ms",                        r"Mean ITL \(ms\):\s*([\d.]+)"),
        ("itl_median_ms",                      r"Median ITL \(ms\):\s*([\d.]+)"),
        ("itl_p99_ms",                         r"P99 ITL \(ms\):\s*([\d.]+)"),
    ]

    # 先确认有 benchmark result 区域
    if "Serving Benchmark Result" not in block:
        return None

    metrics = {}
    for field, pattern in metric_patterns:
        m = re.search(pattern, block)
        metrics[field] = m.group(1) if m else ""

    return metrics


def parse_log_file(log_path: str) -> list[dict]:
    """解析单个 log 文件，返回所有测试记录列表。"""
    log_path = Path(log_path)
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    header = extract_header_info(content)
    tests = extract_test_blocks(content)

    if not tests:
        print(f"  ⚠  {log_path.name}: 未找到有效的测试结果")
        return []

    # 合并头部信息和文件名到每条记录
    for record in tests:
        record["log_file"] = log_path.name
        record.update(header)

    print(f"  ✅ {log_path.name}: 提取到 {len(tests)} 条测试记录")
    return tests


def write_csv(records: list[dict], output_path: str, append: bool = False):
    """将记录写入 CSV 文件。"""
    output_path = Path(output_path)
    mode = "a" if append else "w"
    write_header = not append or not output_path.exists() or output_path.stat().st_size == 0

    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for record in records:
            # 只写 CSV_FIELDS 中定义的列，忽略多余字段
            row = {k: record.get(k, "") for k in CSV_FIELDS}
            writer.writerow(row)


def collect_log_files(input_path: str) -> list[Path]:
    """收集要处理的 log 文件列表。"""
    p = Path(input_path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        files = sorted(p.glob("vllm_bench_*.log"))
        if not files:
            # 放宽匹配：目录下所有 .log 文件
            files = sorted(p.glob("*.log"))
        return files
    else:
        # 尝试 glob 模式
        parent = p.parent if p.parent != p else Path(".")
        files = sorted(parent.glob(p.name))
        return files


def default_output_path(input_path: str) -> str:
    """根据输入路径生成默认的 CSV 输出路径。"""
    p = Path(input_path)
    if p.is_file():
        return str(p.with_suffix(".csv"))
    elif p.is_dir():
        return str(p / "vllm_bench_results.csv")
    else:
        return str(Path(input_path).parent / "vllm_bench_results.csv")


def main():
    parser = argparse.ArgumentParser(
        description="解析 VLLM benchmark 日志并输出 CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python parse_vllm_bench.py vllm_bench_xxx.log
  python parse_vllm_bench.py ./logs/
  python parse_vllm_bench.py vllm_bench_xxx.log -o results.csv
  python parse_vllm_bench.py vllm_bench_xxx.log -o results.csv --append
        """,
    )
    parser.add_argument(
        "input",
        help="输入 log 文件路径或包含 log 文件的目录",
    )
    parser.add_argument(
        "-o", "--output",
        help="输出 CSV 文件路径（默认与 log 文件同名 .csv 或目录下 vllm_bench_results.csv）",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加模式：将结果追加到已有 CSV 文件",
    )

    args = parser.parse_args()

    # 收集要处理的文件
    log_files = collect_log_files(args.input)
    if not log_files:
        print(f"❌ 未找到任何 log 文件: {args.input}")
        sys.exit(1)

    print(f"📂 找到 {len(log_files)} 个 log 文件")

    # 解析所有文件
    all_records = []
    for log_file in log_files:
        records = parse_log_file(str(log_file))
        all_records.extend(records)

    if not all_records:
        print("❌ 所有文件中均未找到有效的测试结果")
        sys.exit(1)

    # 输出 CSV
    output_path = args.output or default_output_path(args.input)
    write_csv(all_records, output_path, append=args.append)

    print(f"\n📊 共 {len(all_records)} 条记录已写入 -> {output_path}")

    # 打印摘要表格
    print("\n" + "=" * 100)
    print(f"{'Test':>5} | {'Prompts':>8} | {'Concurrency':>11} | {'Out tok/s':>10} | "
          f"{'Total tok/s':>11} | {'TTFT Mean':>10} | {'TPOT Mean':>10} | {'ITL Mean':>10}")
    print("-" * 100)
    for r in all_records:
        print(
            f"{r.get('test_id', ''):>5} | "
            f"{r.get('num_prompts', ''):>8} | "
            f"{r.get('max_concurrency', ''):>11} | "
            f"{r.get('output_token_throughput_tok_s', ''):>10} | "
            f"{r.get('total_token_throughput_tok_s', ''):>11} | "
            f"{r.get('ttft_mean_ms', ''):>10} | "
            f"{r.get('tpot_mean_ms', ''):>10} | "
            f"{r.get('itl_mean_ms', ''):>10}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()