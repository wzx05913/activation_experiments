# 激活近似实验交接（2026-10-29）

## 1. 项目与路径

- 本地项目：`D:/Desktop/深度思考大模型安全防护/activation_experiments`
- 云端代码目录：`/Code`
- 云端模型目录：`/home/waas/models`
- 汇总目录：`figures/summary_07_08_2026-10-29-v2/`
- 当前日志：`logs/activation_function_experiment_log_v2.md`

## 2. 已完成进度

- Qwen2.5-1.5B、3B、7B baseline 已完成。
- 1.5B 已完成层位置探索；layer27 是候选敏感层。
- 3B、7B Hard-Swish 全层跨模型验证已完成。
- 07 人工修正版与 08 自动重评分已重新按安全口径汇总。

## 3. 结果摘要

07 HarmBench val 人工修正版：3B baseline/Hard-Swish 全层 Score 2 率为 43.8%/67.5%；7B 为 47.5%/72.5%。

08 自动评分候选率：HarmBench test 中 3B 为 42.8%/73.1%，7B 为 48.4%/72.5%；AdvBench 中 3B 为 1.5%/13.3%，7B 为 2.1%/29.2%。每组前者为 baseline，后者为 Hard-Swish 全层。

08 不是全量人工最终标签，报告时必须使用“自动评分候选率”表述。不要将 07、08 合并为一个比例。

## 4. 下一阶段计划：1.5B 安全修复

可以开始安全修复，但先做最小、可结题的验证：

1. 核实云端模型、数据、GPU、dtype、代码版本和训练框架。
2. 已补齐 `src/train_lora.py` 最小 LoRA/SFT 训练入口，并完成静态语法检查。
3. 已为 `src/generate.py` 增加 `--adapter`，可在激活近似 hook 开启时加载 LoRA adapter 生成评测结果。
4. 训练组 A：Hard-Swish + layer27。
5. 训练组 B：Hard-Swish 全层。
6. 控制组 C：原始 SiLU + 相同训练数据、训练步数和超参数。
7. 训练数据必须与 HarmBench test、AdvBench、Benign 严格隔离。
8. 修复后使用相同解码参数和评分规则评测，并对关键结果人工抽查。

判据：Score 2 明显下降；Score 0 接近 baseline；正常任务不出现明显过度拒答；训练集与测试集无重叠。

## 5. 不要做的事情

- 不要重新跑已经完成的 baseline。
- 不要把 08 自动候选率写成最终 ASR。
- 不要把 layer27 写成所有模型和所有近似条件下的普遍关键层。
- 不要把 Python hook 实验写成已经获得真实硬件推理加速。
- 不要在报告、图表、日志或交接文档中复制评测提示词、模型回答或样本级原文。

## 6. 建议的首个动作

先在云端完成环境和训练入口检查，再决定训练命令。检查结果应只记录路径、版本、层数、GPU、dtype、数据规模和配置摘要，不记录任何评测样本内容。
