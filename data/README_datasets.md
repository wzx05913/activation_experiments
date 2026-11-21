# 数据集使用说明

## 当前已下载文件

| 文件 | 数量 | 用途 | 当前状态 |
|---|---:|---|---|
| `advbench/harmful_behaviors.csv` | 520 | 直接有害请求，观察模型是否生成不安全回答 | 可直接运行，使用 `goal` 列 |
| `advbench/harmful_strings.csv` | 574 | 有害字符串/目标文本，主要用于攻击或分类实验 | 当前主实验暂不使用 |
| `harmbench/harmbench_behaviors_text_val.csv` | 80 | HarmBench 验证集，快速检查流程 | 可直接运行，使用 `Behavior` 列 |
| `harmbench/harmbench_behaviors_text_test.csv` | 320 | HarmBench 正式测试集 | 可直接运行，使用 `Behavior` 列 |
| `xstest/xstest_prompts.csv` | 450 | 安全边界测试：250 safe、200 unsafe | 可直接运行，使用 `prompt` 列 |
| `benign_utility.jsonl` | 50 | 普通正常任务效用评估 | 本项目建立的 benign 集 |

## 建议的正式实验数据集

第一阶段使用 4 个数据集：

1. `benign_utility.jsonl`：正常任务能力和效用；
2. `harmbench_behaviors_text_test.csv`：主要安全退化指标；
3. `advbench/harmful_behaviors.csv`：直接有害请求的补充验证；
4. `xstest/xstest_prompts.csv`：安全边界、正确拒答和过度拒答。

`harmbench_behaviors_text_val.csv` 只用于开发阶段和小规模试跑；确定代码无误后，正式表格使用 test 集。`harmful_strings.csv` 暂不作为主评测输入，因为它是目标字符串集合，不是普通用户请求集合。

## 命令行输入规则

`src/generate.py` 现在支持 `.jsonl` 和 `.csv`：

- AdvBench CSV 自动读取 `goal`；
- HarmBench CSV/JSONL 使用 `Behavior`；有 `ContextString` 时输入为 `ContextString + "\n\n---\n\n" + Behavior`，只拼接一次；
- XSTest CSV 自动读取 `prompt`；
- 旧版 AdvBench JSONL 如果 `prompt` 为空，会自动回退到 `raw.content[0]`。

所有输入都会被统一转换为带有 `prompt` 字段的内部记录。参考答案列不会被作为模型输入。


## 2026-10-28 输入协议修正

- 支持 CSV、JSONL 以及 JSON 对象/对象数组；JSONL 和 CSV 使用相同字段回退规则。
- 上下文拼接保持幂等；识别早期 Context/Task 单次或重复包装并转成统一形式；遇到含上下文但自定义 prompt 的歧义输入明确报错，避免静默丢弃内容。
- 生成前检查全部聊天格式输入的 token 长度，不再静默截断；`--check-inputs-only` 可仅用 tokenizer 检查，不加载模型权重。
- 新输出记录输入协议、上下文注入标记、输入/输出 token 数和生成参数。达到输出上限只标为“可能截断”，不能据此认定任务完成。
- 旧结果缺失上下文不能通过重新评分补救。修正输入后的 baseline 与近似结果应存入新目录，参数一致。
