# 07/08 结果汇总（2026-10-29）

## 统计口径

- 07 使用 `results/07_context_v2/manual_review_corrected_v1/`：以原自动评分为底，对 `correction_log.csv` 中的人工修正逐条覆盖。该批为 HarmBench val。
- 08 使用 `results/08_test_adv/rescored_v5_final/`：使用 `screen_v5_2026-10-29` 自动辅助评分，包含 HarmBench test 和 AdvBench。
- Score 2 在本表中作为“有害输出候选/安全失调候选”统计，不直接等同于最终人工 ASR。

## 汇总表

|批次|模型|数据集|设置|N|Score 0|Score 1|Score 2|Score 2率|修正条数|标签口径|
|-|-|-|-:|-:|-:|-:|-:|-:|-|
|07_context_v2_manual_corrected_v1|qwen25_3b|HarmBench val|baseline|80|34|10|36|45.0%|9|人工修正覆盖自动评分|
|07_context_v2_manual_corrected_v1|qwen25_3b|HarmBench val|hard_swish_all|80|16|5|59|73.8%|5|人工修正覆盖自动评分|
|07_context_v2_manual_corrected_v1|qwen25_7b|HarmBench val|baseline|80|29|10|41|51.2%|9|人工修正覆盖自动评分|
|07_context_v2_manual_corrected_v1|qwen25_7b|HarmBench val|hard_swish_all|80|14|1|65|81.2%|9|人工修正覆盖自动评分|
|08_test_adv_rescored_v5_final|qwen25_3b|AdvBench|baseline|520|512|0|8|1.5%|0|screen_v5 自动辅助评分|
|08_test_adv_rescored_v5_final|qwen25_3b|HarmBench test|baseline|320|183|0|137|42.8%|0|screen_v5 自动辅助评分|
|08_test_adv_rescored_v5_final|qwen25_3b|AdvBench|hard_swish_all|520|451|0|69|13.3%|0|screen_v5 自动辅助评分|
|08_test_adv_rescored_v5_final|qwen25_3b|HarmBench test|hard_swish_all|320|85|1|234|73.1%|0|screen_v5 自动辅助评分|
|08_test_adv_rescored_v5_final|qwen25_7b|AdvBench|baseline|520|509|0|11|2.1%|0|screen_v5 自动辅助评分|
|08_test_adv_rescored_v5_final|qwen25_7b|HarmBench test|baseline|320|165|0|155|48.4%|0|screen_v5 自动辅助评分|
|08_test_adv_rescored_v5_final|qwen25_7b|AdvBench|hard_swish_all|520|367|1|152|29.2%|0|screen_v5 自动辅助评分|
|08_test_adv_rescored_v5_final|qwen25_7b|HarmBench test|hard_swish_all|320|87|1|232|72.5%|0|screen_v5 自动辅助评分|

## 主要趋势

- 07 的 1.5B/3B/7B HarmBench val 结果显示，Hard-Swish 全层的 Score 2 率高于对应 baseline；该趋势支持继续开展安全修复，但 07 是 validation 集。
- 08 的 3B/7B HarmBench test 与 AdvBench 结果中，Hard-Swish 全层的 Score 2 率均高于 baseline；跨模型方向一致。
- 08 的数值仍是自动辅助筛选结果，Score 2 应称为候选比例；修复前后应使用同一规则、同一数据集和同一解码参数比较。
- 下一步可以在 1.5B 上进入轻量安全对齐修复，同时保留原始 SiLU 与 Hard-Swish 未修复模型作为对照。

## 文件

- 汇总 CSV：`figures/summary_07_08_2026-10-29/summary_07_08.csv`
- 图表说明：`figures/summary_07_08_2026-10-29/README.md`