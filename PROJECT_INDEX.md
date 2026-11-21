# 项目文件索引

从 [项目总入口](../README.md) 或 [实验总结日志](logs/experiment_summary_2026-11-22.md) 开始阅读。本索引按用途归档，不移动原始文件。

## 工作区材料

| 目录 | 用途 |
|---|---|
| [申报书和中期](../申报书和中期/) | 项目申请和中期/项目材料；原始承诺应与实测完成项分别表述 |
| [结题要求](../结题要求/) | 学院答辩与推优说明、系统操作说明、结题验收书模板；国创年会材料单独作为参考 |
| [相关论文](../相关论文/) | 方法背景与参考材料，不是本项目的实验产物 |
| [activation_experiments](README.md) | 实验代码、配置、数据、评测、图表、日志和交接 |
| `tmp/pdfs/` | 临时提取与阅读材料，不作为原始通知或最终成果 |

## 实验代码与资产

| 路径 | 内容 |
|---|---|
| [src](src/) | `generate.py` 生成；`activation_hooks.py` / `approximation.py` 注入扰动和近似；`score_results.py` / `review_results.py` / `summarize_results.py` 评分、复核与汇总 |
| [训练和诊断](src/train_dpo.py) | `train_dpo.py` 训练、`activation_regularizer.py` 表征约束、`evaluate_dpo_holdout.py` 偏好留出诊断；`prepare_*`、`curate_*`、`audit_*` 等用于数据准备与审核 |
| [scripts](scripts/) | 批量训练与评测入口、分层筛查、对照汇总及绘图辅助脚本 |
| [configs](configs/) | 基线、层 0 Gaussian/稀疏化、三种 `silu_*` 全层设置；具体运行以日志、命令和产物元数据为准 |
| [tests](tests/) | 激活 hook、表征正则、数据处理和评测流程的测试 |
| [data](data/) | [评测数据说明](data/README_datasets.md)、[PKU 数据说明](data/README_PKU_SafeRLHF.md)、派生训练集、辅助样例和缓存；不将辅助样例计入正式四套评测 |
| [figures](figures/) | 结题可用图表和配套结果表，引用时保留数据划分与评分口径 |
| [依赖](requirements.txt) | Python 依赖；`.venv/`、`.idea/`、`__pycache__/` 和 `.lock` 是环境或缓存，不作为结题成果 |
| 根层辅助文件 | `fix_order.py`、`make_comparison.py`、`patch_charts.py`、`patch_stacked.py`、`patch_svg.py` 用于历史汇总或图表处理；`manual_review_dump.json` 是复核辅助记录，`tips.txt` 是备忘 |

## 结果目录

| 路径 | 对应阶段 |
|---|---|
| [01_baseline-1.5b](results/01_baseline-1.5b/) | 1.5B 早期原始基线 |
| [02_smoke_tests](results/02_smoke_tests/) | 流程小样本验证，不作正式效果结论 |
| [03_summaries](results/03_summaries/) | 早期汇总 |
| [04_approximation](results/04_approximation/) | MLP 输出扰动和模拟量化 |
| [05_silu_approximation](results/05_silu_approximation/) | MLP 内部 SiLU 替换，多种近似对比 |
| [06_cross_model](results/06_cross_model/) | 跨模型实验 |
| [07_context_v2](results/07_context_v2/) | 更新提示口径后的验证与复核 |
| [08_test_adv](results/08_test_adv/) | HarmBench test、AdvBench、单层和全层对照 |
| [09_repair](results/09_repair/) | DPO、全层/6-bit 补充、正则消融、adapter 和诊断；各子目录留存程度不同 |

## 历史日志与交接

| 记录 | 在总结中的作用 |
|---|---|
| [10-28 激活实验交接](logs/activation_experiment_handoff_2026-10-28.md) | 早期基线、扰动和评测流程 |
| [10-29 激活实验交接](logs/activation_experiment_handoff_2026-10-29.md) | 近似、跨模型和敏感层主线 |
| [10-31 PKU 拒答筛选](logs/pku_refusal_screening_2026-10-31.md) | 训练样本筛选记录 |
| [10-31 WildJailbreak 运行手册](logs/wildjailbreak_repair_runbook_2026-10-31.md) | 备选修复流程，不算已验证修复 |
| [11-02 索引 27 DPO 手册](logs/pku_dpo_layer27_runbook_2026-11-02.md) | 单层修复协议和入口 |
| [11-02 混合修复手册](logs/pku_mixed_repair_runbook_2026-11-02.md) | mixed SFT 备选流程 |
| [11-03 DPO v6 小规模集](logs/pku_dpo_curated_pilot_v6_2026-11-03.md) | 偏好对人工筛选 |
| [11-03 DPO 扩充覆盖](logs/pku_dpo_expansion_coverage_2026-11-03.md) | 数据扩充与覆盖核对 |
| [11-03 DPO 覆盖清理](logs/pku_dpo_risk_coverage_cleanup_2026-11-03.md) | 数据清理与复审 |
| [11-03 DPO v6 诊断](logs/pku_dpo_v6_diagnostic_runbook_2026-11-03.md) | 偏好留出与诊断流程 |
| [11-05 67+7 修复交接](handoff/repair_v6_plus7_2026-11-05.md) | 74 对训练集形成与跟进 |
| [11-10 修复状态](handoff/repair_status_2026-11-10.md) | 当时状态；epoch 3 test 后由 11-11 对照补齐 |
| [11-16 修复综述](logs/repair_summary_2026-11-16.md) | 单层主结果、全层与 3B 修复边界 |
| [11-17 量化修复交接](handoff/quantize_repair_handoff_2026-11-17.md) | 6-bit 补充实验及当时归档情况 |
| [11-19 表征正则交接](handoff/activation_regularizer_runbook_2026-11-19.md) | 正则实现、运行、结果与复核注意事项 |

历史记录保留当时状态，新总结以更新后的结果和本地实存文件为准。原始输出和训练对可能包含敏感内容，结题材料优先使用统计表、图表与脱敏样例，不直接公开原始明细。
