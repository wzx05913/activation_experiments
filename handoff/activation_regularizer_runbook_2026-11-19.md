# 第 27 层表征正则实验

本版本是固定 Hard-Swish 近似下的 DPO 加有害提示聚集正则，不是完整 QuadA。论文正则位于第一层；这里使用索引 27 的 Transformer block 输出，位于最终模型归一化之前。近似层与正则层是独立参数。

## 目标与实现

目标为 `L_DPO + weight * (1 - mean_off_diagonal_cosine)`。对不同有害提示的表示计算余弦相似度，不含自身配对。只输入聊天模板及提示，取最后一个有效输入 token，不包含 chosen/rejected、EOS 回答或 padding。计算余弦时使用 float32。

沿用原有 DPO 梯度累积，每次 optimizer 更新前额外进行一次提示批次前向与正则反传。默认从当前累积组随机选 2 条，最大可通过 `--activation-reg-batch-size` 修改；不足时只用该组实际大小。若尾组只有 1 条，从训练集补 1 条不同提示。采样使用独立 RNG，不改变 DPO 样本顺序。74 对、累积 8 时每轮 10 次更新，每次正则只加一次，不再除以累积步数。

启用时必须确认训练提示全部为有害请求，使用 `--activation-reg-harmful-only` 声明。此开关只是数据语义声明，不能替代审核；混合正常请求的数据不可直接用于聚集。独立 holdout 和评测集不会进入正则采样。

正则前向暂时关闭 gradient checkpointing，确保 block hook 捕获的是带梯度张量，退出后恢复现有默认 checkpointing 设置。DPO 前向仍使用 checkpointing。额外前向可能增加显存与耗时，默认先用 2 条；本地小模型测试不代表 1.5B GPU 已验证。

## 云端运行

同步 `src/train_dpo.py`、新增的 `src/activation_regularizer.py` 和 `scripts/run_1p5b_dpo_activation_reg.sh` 到云端对应位置。云端根目录沿用 `/Code`，模型路径可用 `MODEL_PATH` 覆盖。脚本要求原始训练文件恰为 74 对，训练程序进一步检查审核标记和重复。

先做 4 对、1 轮功能检查，输出到独立目录：

```bash
cd /Code
TRAIN_EPOCHS=1 TRAIN_MAX_PAIRS=4 bash scripts/run_1p5b_dpo_activation_reg.sh results/09_repair/activation_reg_smoke_v1 0.1
```

成功后从同一原始模型分别运行普通 DPO 与正则实验，不加载旧 adapter，不把冒烟检查作为正式结果：

```bash
bash scripts/run_1p5b_dpo_activation_reg.sh results/09_repair/layer27_dpo_reg_control_v1 0
bash scripts/run_1p5b_dpo_activation_reg.sh results/09_repair/layer27_dpo_reg_w010_v1 0.1
```

如需验证正则权重是否过小，可从同一原始模型补跑一组 `0.5`。该组只改变正则权重，仍使用 74 对数据、3 epoch、索引 27 层、seed 42；不要从 `0.1` 的 adapter 继续训练：

```bash
bash scripts/run_1p5b_dpo_activation_reg.sh results/09_repair/layer27_dpo_reg_w050_v1 0.5
```

同步更新后的 `scripts/eval_1p5b_dpo_activation_reg.sh`，训练完成后只评测 `0.5` 组，复用现有普通 DPO 和 `0.1` 组结果：

```bash
REG_ADAPTER=results/09_repair/layer27_dpo_reg_w050_v1 \
bash scripts/eval_1p5b_dpo_activation_reg.sh results/09_repair/layer27_dpo_reg_w050_eval_v1 --regularized-only
```

这次运行的目的只是检验权重敏感性，不应根据训练集 preference accuracy 单独判断收益。最终仍以相同设置下的 HarmBench 人工复核和 XSTest `safe` 误拒答比较为准；若 `0.5` 没有清晰改善，就停止扩大权重搜索。

两组均默认 74 对、3 epoch、学习率 1e-4、LoRA r=4/alpha=8、beta=0.1、sum log-prob、seed 42。原始训练实际配置应与云端旧 `repair_config.json` 核对；特别是 token 长度限制和运行环境。脚本不覆盖已有非空输出。权重 0 不运行额外前向、采样或梯度探针。

## 观察与验收

- 首次正则输出 `activation_reg_gradient_norm`，是独立正则对可训练参数的梯度范数；为零或非有限值直接报错。梯度探针不修改已有 DPO `.grad`。
- `diagnostic_history.json` 每轮包含 DPO loss，以及 `activation_regularizer` 的原始/加权 loss、平均 cosine、更新次数、实际批大小。它们是在线训练批次统计，不能当成固定集合的泛化或聚类诊断。
- `repair_config.json` 保存正则参数、提取位置、采样协议和首次梯度范数。
- 用一致生成设置评估 HarmBench val、XSTest safe/unsafe、benign，区分自动候选与审核结论。先验收安全与正常请求效用，不按已有 test 反复调参。
- 正则只拉近有害提示，没有显式分离正常提示；余弦升高不等于安全改善。固定正常/有害提示的跨组表示分析尚未自动实现，可作为后续诊断，不能用当前训练曲线宣称已证明分离。
- 若效用明显恶化或安全无改善，保留普通 DPO 主结果，记录这一有界消融即可。0.1 是预设起点，不是已验证的最佳值。

## 本地检查

在 `activation_experiments` 下执行：

```bash
python -m unittest discover -s tests -p 'test_activation_regularizer.py' -v
python -m unittest discover -s tests -p 'test_activation_hooks_wrapped.py' -v
```
