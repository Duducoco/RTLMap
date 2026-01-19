#!/bin/bash
# CV32E40P 模块 CDFG 标注测试命令
# 按节点数从小到大排序（共24个模块，21个有覆盖率数据）
# 生成时间: 2026-01-16

DESIGN_NAME="cv32e40p"
COVERAGE_PATH="designs/cv32e40p/coverage_reports/coverage_report1"

echo "=========================================="
echo "CV32E40P CDFG 标注测试"
echo "=========================================="

# 1. cv32e40p_sleep_unit (33节点, 36边) ✅ 有覆盖率
echo "[1/24] cv32e40p_sleep_unit (33节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_sleep_unit -v --svg

# 2. cv32e40p_popcnt (34节点, 125边) ✅ 有覆盖率
echo "[2/24] cv32e40p_popcnt (34节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_popcnt -v --svg

# 3. cv32e40p_hwloop_regs (40节点, 89边) ❌ 无覆盖率 - 跳过
echo "[3/24] cv32e40p_hwloop_regs (40节点) - 跳过: 无覆盖率数据"
# uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_hwloop_regs -v --svg

# 4. cv32e40p_obi_interface (49节点, 84边) ✅ 有覆盖率
echo "[4/24] cv32e40p_obi_interface (49节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_obi_interface -v --svg

# 5. cv32e40p_int_controller (54节点, 149边) ✅ 有覆盖率
echo "[5/24] cv32e40p_int_controller (54节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_int_controller -v --svg

# 6. cv32e40p_fifo (61节点, 173边) ✅ 有覆盖率
echo "[6/24] cv32e40p_fifo (61节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_fifo -v --svg

# 7. cv32e40p_ff_one (68节点, 188边) ✅ 有覆盖率
echo "[7/24] cv32e40p_ff_one (68节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_ff_one -v --svg

# 8. cv32e40p_alu_div (77节点, 187边) ✅ 有覆盖率
echo "[8/24] cv32e40p_alu_div (77节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_alu_div -v --svg

# 9. cv32e40p_aligner (93节点, 262边) ✅ 有覆盖率
echo "[9/24] cv32e40p_aligner (93节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_aligner -v --svg

# 10. cv32e40p_prefetch_controller (95节点, 200边) ✅ 有覆盖率
echo "[10/24] cv32e40p_prefetch_controller (95节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_prefetch_controller -v --svg

# 11. cv32e40p_mult (122节点, 291边) ✅ 有覆盖率
echo "[11/24] cv32e40p_mult (122节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_mult -v --svg

# 12. cv32e40p_apu_disp (138节点, 262边) ❌ 无覆盖率 - 跳过
echo "[12/24] cv32e40p_apu_disp (138节点) - 跳过: 无覆盖率数据"
# uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_apu_disp -v --svg

# 13. cv32e40p_compressed_decoder (143节点, 476边) ✅ 有覆盖率
echo "[13/24] cv32e40p_compressed_decoder (143节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_compressed_decoder -v --svg

# 14. cv32e40p_prefetch_buffer (146节点, 362边) ✅ 有覆盖率
echo "[14/24] cv32e40p_prefetch_buffer (146节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_prefetch_buffer -v --svg

# 15. cv32e40p_load_store_unit (166节点, 438边) ✅ 有覆盖率
echo "[15/24] cv32e40p_load_store_unit (166节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_load_store_unit -v --svg

# 16. cv32e40p_if_stage (401节点, 1180边) ✅ 有覆盖率
echo "[16/24] cv32e40p_if_stage (401节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_if_stage -v --svg

# 17. cv32e40p_cs_registers (457节点, 1390边) ✅ 有覆盖率
echo "[17/24] cv32e40p_cs_registers (457节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_cs_registers -v --svg

# 18. cv32e40p_alu (502节点, 1548边) ✅ 有覆盖率
echo "[18/24] cv32e40p_alu (502节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_alu -v --svg

# 19. cv32e40p_ex_stage (688节点, 1909边) ✅ 有覆盖率
echo "[19/24] cv32e40p_ex_stage (688节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_ex_stage -v --svg

# 20. cv32e40p_controller (755节点, 2184边) ✅ 有覆盖率
echo "[20/24] cv32e40p_controller (755节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_controller -v --svg

# 21. cv32e40p_decoder (1737节点, 6210边) ✅ 有覆盖率
echo "[21/24] cv32e40p_decoder (1737节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_decoder -v --svg

# 22. cv32e40p_top (3245节点, 10905边) ❌ 无覆盖率 - 跳过
echo "[22/24] cv32e40p_top (3245节点) - 跳过: 无覆盖率数据"
# uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_top -v --svg

# 23. cv32e40p_core (3255节点, 10914边) ✅ 有覆盖率
echo "[23/24] cv32e40p_core (3255节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_core -v --svg

# 24. cv32e40p_id_stage (3389节点, 11425边) ✅ 有覆盖率
echo "[24/24] cv32e40p_id_stage (3389节点)"
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_id_stage -v --svg

echo "=========================================="
echo "测试完成!"
echo "=========================================="
uv run python data_annotate.py --design_name cv32e40p --module_name cv32e40p_register_file_ff -v