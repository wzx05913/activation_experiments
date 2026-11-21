# 激活函数近似主线已切换

当前主线不再把 MLP 输出上的 Gaussian/Laplace/Quantize 作为“激活函数近似”。新的实验代码直接替换 Qwen/Llama 风格 MLP 内部的 `act_fn`，优先面向 Qwen2.5 的 SwiGLU 中 `SiLU`。

新主线配置：

- `configs/silu_hard_swish_all.yaml`
- `configs/silu_piecewise_linear_all.yaml`
- `configs/silu_piecewise_polynomial_all.yaml`
- `configs/baseline.yaml`

运行入口仍是 `src.generate`。`--layers all` 对所有 decoder layer 替换；显式层号用于敏感性分析。旧 `gaussian`、`laplace`、`sparsify`、`quantize` 参数保留，仅用于复现历史结果，并在输出中标记为 `mlp_output_legacy`。

先运行函数级误差分析：

```powershell
cd activation_experiments
.\\.venv\\Scripts\\python.exe -m src.analyze_silu_approximation --output <writable-output>\\function_error.json
```

再用 Benign 小样本做 smoke test。注意：当前 Python/PyTorch 替换器用于验证安全影响，不自动提供真实硬件加速；只有融合 kernel 或目标推理框架的实测才能宣称 wall-clock speedup。

早期研究路线见 [实验 README](README.md)，实际完成情况与证据见 [实验总结日志](logs/experiment_summary_2026-11-22.md)。
