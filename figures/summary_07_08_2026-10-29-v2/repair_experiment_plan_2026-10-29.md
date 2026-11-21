# 1.5B 激活近似安全修复实验计划（2026-10-29）

## 一、目标

验证在 Qwen2.5-1.5B 上，保持 Hard-Swish 激活近似开启时，轻量 LoRA/SFT 是否能把安全表现拉回原始 SiLU 基线附近，同时不明显损害 benign 能力。

## 二、总体路线：两阶段，不直接复现完整 QuadA

本项目以 Hard-Swish 作为当前激活近似对象。修复目标首先是恢复实际 Hard-Swish 部署条件下的安全性，而不是直接声称解决所有激活近似问题。

- 阶段一（主线）：Hard-Swish 条件下使用 LoRA + SFT。
- 阶段二（扩展）：Hard-Swish 保持开启，并在已识别的敏感层加入小幅激活扰动，比较普通 Hard-Swish-SFT 与扰动增强 SFT。

QuadA 的 MVA 是用于训练的最脆弱近似扰动，不等同于敏感层本身，也不等同于 Hard-Swish。当前主线优先对齐实际部署的 Hard-Swish；MVA、DPO 和激活余弦正则不作为首轮必做项。

## 三、阶段一首轮只做四组

| 组别 | 训练状态 | 训练激活条件 | 测试激活条件 | 目的 |
|---|---|---|---|---|
| A | 不训练 | - | Hard-Swish layer27 | 该条件下的退化基线 |
| B | 不训练 | - | Hard-Swish all layers | 全层退化基线 |
| C | LoRA + SFT | SiLU | Hard-Swish layer27 | 普通安全微调的迁移控制 |
| D | LoRA + SFT | Hard-Swish layer27 | Hard-Swish layer27 | 近似感知修复主组 |

实际执行时，C 用于迁移控制，D 是主结论组；如果算力有限，优先完成 C+D 的配对比较，再补全层 B。不要在首轮同时展开 rank、epoch、层组合和近似函数的大规模网格。

核心比较是：SiLU 条件 SFT 后测试 Hard-Swish，与 Hard-Swish 条件 SFT 后测试 Hard-Swish。

## 四、训练数据格式与构成

阶段一使用普通安全 SFT，不使用 DPO。建议初始构成为：60% 明确有害请求-安全拒答，20% 双用途/边界请求，20% 正常请求。首轮约 250 条做流程试跑，再扩展到 500--1000 条。训练集必须与 HarmBench test、AdvBench、XSTest 和 benign 评测集按 prompt 去重隔离。

## 五、阶段二：可选的 QuadA-inspired 扰动增强

在阶段一主组稳定后，再测试 Hard-Swish + 敏感层小幅扰动 + LoRA/SFT。该组用于验证对近似误差变化的鲁棒性，不是完整 MVA-DPO，也不应写成 QuadA 复现。

## 六、运行前检查

训练数据必须替换 `data/safety_train.jsonl` 和 `data/safety_test.jsonl` 中的占位内容，并确保与 HarmBench test、AdvBench、Benign、XSTest 隔离。先运行：

```powershell
python -m activation_experiments.src.audit_repair_data `
  --train activation_experiments/data/safety_train.jsonl `
  --test activation_experiments/data/safety_test.jsonl
```

当前本地环境没有 `torch`、`transformers`、`peft`，所以本地只完成语法检查；正式训练需在云端环境完成依赖、模型路径和 GPU 检查。

## 七、训练命令模板

```powershell
python -m activation_experiments.src.train_lora `
  --model /path/to/Qwen2.5-1.5B-Instruct `
  --train-data activation_experiments/data/safety_train.jsonl `
  --output-dir activation_experiments/results/09_repair/qwen25_1p5b/hard_swish_layer27_lora `
  --approximation hard_swish --layers 27 `
  --epochs 2 --batch-size 1 --gradient-accumulation-steps 8 `
  --learning-rate 2e-4 --lora-r 8 --lora-alpha 16 `
  --device cuda --dtype bfloat16
```

B 组改为 `--layers all`；C 组改为 `--approximation none --layers 27`（近似关闭时层参数不会生效），并使用同一训练数据、seed、步数和超参数。

## 八、修复后生成

```powershell
python -m activation_experiments.src.generate `
  --model /path/to/Qwen2.5-1.5B-Instruct `
  --adapter activation_experiments/results/09_repair/qwen25_1p5b/hard_swish_layer27_lora `
  --data /path/to/eval.jsonl `
  --output /path/to/repair_layer27.jsonl `
  --setting repair_layer27 `
  --approximation hard_swish --layers 27 `
  --max-input-tokens 1024 --max-new-tokens 128 `
  --temperature 0 --top-p 1 --device cuda --dtype bfloat16
```

B 组使用 `--layers all`，C 组去掉 `--adapter`、`--approximation` 和 `--layers`，但保持其他生成参数完全一致。

## 九、论文参考与采用方式

1. **Activation Approximations Can Incur Safety Vulnerabilities Even in Aligned LLMs: Comprehensive Analysis and Defense**：项目内 `相关论文/激活近似/MinerU_markdown_Activation_Approximations__2107429045854089216.md`。最直接参考。QuadA 的核心包括近似感知安全对齐、MVA 最坏情况扰动和安全敏感层选择。当前项目采用其简化版：固定 Hard-Swish、固定 layer27、LoRA/SFT；不声称完整复现 QuadA。
2. **Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models**：申报书参考文献[18]，项目内量化论文 PDF。可借鉴“高精度安全模型/退化模型构造修复数据、LoRA 轻量补丁、选择性更新”的思路，但它针对量化，不应称为激活近似方法。
3. **LoRA: Low-Rank Adaptation of Large Language Models**：申报书参考文献[4]。用于支撑低成本参数高效微调选择。
4. **Direct Preference Optimization (DPO)**：申报书将其作为偏好修复损失依据；本轮不实现 DPO，先用可控的 SFT 完成最小闭环，避免数据构造和训练复杂度过高。
5. **Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs**：项目内激活近似论文目录。它提醒修复训练必须加入 benign、XSTest/过度拒答检查，并保持训练集与评测集隔离；不是本项目修复算法的直接基线。

## 十、结题判据

- HarmBench/AdvBench 的 Score 2 或自动候选率相对未修复 Hard-Swish 明显下降；
- 尽量接近原始 SiLU baseline；
- benign 不出现明显过度拒答；
- XSTest 不出现明显边界恶化；
- 报告中区分人工修正版与自动评分候选率，不把候选率直接写成最终 ASR；
- 记录模型、层、近似函数、数据规模、seed、dtype、生成参数和代码版本。
