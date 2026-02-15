#!/usr/bin/env python3
"""RTLIL JSON 提取脚本

独立于三阶段流水线，预先为指定 design 生成 RTLIL JSON 文件。
从 datamodule.py 中提取的 RTLIL 生成逻辑，可单独运行以提前准备数据。

用法:
    python -m scripts.RTLIL_extract \
        --sim-results-dir designs/cv32e40p/simulation_results \
        --module-names cv32e40p_int_controller cv32e40p_decoder

    # 不指定 module_names 则跳过按模块生成，仅确保目录存在
    python -m scripts.RTLIL_extract \
        --sim-results-dir designs/cv32e40p/simulation_results
"""

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import YosysRunner

logger = logging.getLogger(__name__)


def extract_rtlil_json(
    sim_results_dir: Path,
    module_names: list[str] | None = None,
) -> list[Path]:
    """为指定 design 生成 RTLIL JSON 文件。

    Args:
        sim_results_dir: simulation_results 目录路径
        module_names: 要生成的模块名列表，None 则仅确保目录存在

    Returns:
        成功生成（或已存在）的 RTLIL JSON 文件路径列表
    """
    design_dir = sim_results_dir.parent
    rtlil_json_dir = design_dir / "RTLIL_json"
    # 确保 RTLIL_json 目录存在
    if not rtlil_json_dir.is_dir():
        try:
            rtlil_json_dir.mkdir(parents=True, exist_ok=True)
            logger.info("已创建 RTLIL JSON 目录: %s", rtlil_json_dir)
        except Exception as e:
            raise RuntimeError(f"无法创建 RTLIL JSON 目录: {rtlil_json_dir}") from e

    generated: list[Path] = []

    if module_names is None:
        logger.info("未指定 module_names，仅确保目录存在: %s", rtlil_json_dir)
        return generated

    flist = design_dir / f"{design_dir.name}.flist"

    for name in module_names:
        json_path = rtlil_json_dir / f"{name}.json"
        if json_path.exists():
            logger.info("RTLIL JSON 已存在，跳过: %s", json_path)
            generated.append(json_path)
            continue

        logger.info("生成 RTLIL JSON: %s", json_path)
        try:
            runner = YosysRunner()
            runner.get_json(
                top_module=name,
                output_dir=rtlil_json_dir,
                flist=flist,
                files=None,
            )
            if json_path.exists():
                generated.append(json_path)
                logger.info("生成成功: %s", json_path)
            else:
                logger.warning("YosysRunner 执行完毕但文件未生成: %s", json_path)
        except Exception as e:
            logger.error("YosysRunner 生成失败 [%s]: %s", name, e)

    return generated


def extract_all_from_sim_results(
    sim_results_dirs: list[str],
    module_names: list[str] | None = None,
) -> list[Path]:
    """批量处理多个 simulation_results 目录。

    Args:
        sim_results_dirs: simulation_results 目录路径列表
        module_names: 要生成的模块名列表

    Returns:
        所有成功生成的 RTLIL JSON 文件路径列表
    """
    all_generated: list[Path] = []

    for raw_path in sim_results_dirs:
        sim_results_dir = Path(raw_path)
        if not sim_results_dir.is_dir():
            logger.warning("simulation_results 目录不存在: %s", sim_results_dir)
            continue

        results = extract_rtlil_json(sim_results_dir, module_names)
        all_generated.extend(results)

    logger.info("共生成 %d 个 RTLIL JSON 文件", len(all_generated))
    return all_generated


def main():
    parser = argparse.ArgumentParser(
        description="RTLIL JSON 提取 - 使用 Yosys 从 RTL 源码生成 RTLIL JSON",
    )
    parser.add_argument(
        "--sim-results-dir",
        type=str,
        nargs="+",
        required=True,
        help="simulation_results 目录路径（可指定多个）",
    )
    parser.add_argument(
        "--module-names",
        type=str,
        nargs="+",
        default=None,
        help="要生成的模块名列表（不含 .json 后缀），不指定则仅确保目录存在",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="启用详细日志输出",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = extract_all_from_sim_results(
        sim_results_dirs=args.sim_results_dir,
        module_names=args.module_names,
    )

    for p in results:
        print(p)


if __name__ == "__main__":
    main()
