# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare current sweep results against the mainline perf cache.

Excludes core attention shapes ("Attn" in Shape) and counts every CSV row
(no deduplication by MxKxN), matching how summarize_metrics.py computes
its "Full model ex core attention" averages.

Usage:
  # First extract the mainline cache:
  git show origin/mainline:test/integration/nkilib/experimental/matmul_mxfp8/mxfp8_perf_cache.json \
    > /tmp/mainline_mxfp8_perf_cache.json

  python compare_to_mainline.py metrics_summary.csv /tmp/mainline_mxfp8_perf_cache.json
"""

import argparse
import csv
import json
from collections import defaultdict


def classify_config(r):
    lhs, swiz = r["lhs_dtype"], r["lhs_is_swizzled"]
    sp, sr = r["enable_scale_packing"], r["spill_reload"]
    if lhs == "MXFP8":
        return "prequantized_swizzled"
    if swiz == "False":
        return "unswizzled_bf16_spill_reload"
    if sp == "True" and sr == "True":
        return "swizzled_bf16_scale_packing_spill_reload"
    if sp == "True":
        return "swizzled_bf16_scale_packing"
    if sr == "True":
        return "swizzled_bf16_spill_reload"
    return "swizzled_bf16_no_scale_packing"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to metrics_summary.csv from current sweep")
    parser.add_argument("mainline_cache", help="Path to mainline mxfp8_perf_cache.json")
    args = parser.parse_args()

    with open(args.mainline_cache) as f:
        prev = json.load(f)
    with open(args.csv_path) as f:
        rows = list(csv.DictReader(f))

    # Current sweep: collect speedups as a list per config (one entry per CSV row)
    cfg_speedups = defaultdict(list)
    for r in rows:
        infer = float(r["InferenceTime (µs)"].replace(" µs", ""))
        if infer < 0:
            continue
        if "Attn" in r["Shape"]:
            continue
        cfg = classify_config(r)
        speedup_str = r["Speedup vs BF16"].replace("x", "")
        if speedup_str == "N/A":
            continue
        cfg_speedups[cfg].append(float(speedup_str))

    # Mainline cache: collect speedups per config, also excluding core attention
    cfg_prev_speedups = defaultdict(list)
    for entry in prev.values():
        shape_name = entry.get("shape_name", "")
        if "Attn" in shape_name:
            continue
        cfg = entry.get("config_type", "")
        sp = entry.get("speedup_vs_bf16")
        if sp:
            cfg_prev_speedups[cfg].append(sp)

    print(
        f"{'Config Type (ex core attention)':<40} | {'This CR Speedup':>15} | {'Mainline Speedup':>16} | {'Δ Speedup Ratio':>15}"
    )
    print("-" * 95)
    for cfg in sorted(cfg_speedups):
        avg = sum(cfg_speedups[cfg]) / len(cfg_speedups[cfg])
        prev_list = cfg_prev_speedups.get(cfg)
        prev_avg = sum(prev_list) / len(prev_list) if prev_list else None
        if prev_avg:
            delta = (avg - prev_avg) / prev_avg * 100
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "N/A (first run)"
        prev_str = f"{prev_avg:.2f}x" if prev_avg else "N/A"
        print(f"{cfg:<40} | {avg:.2f}x           | {prev_str:<16} | {delta_str}")


if __name__ == "__main__":
    main()
