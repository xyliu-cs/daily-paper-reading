# 每日论文报告 — AutoHarness: 通过自动合成代码约束装置改进 LLM 智能体

## 结构化摘要

| 维度 | 内容 |
|---|---|
| **标题** | AutoHarness: improving LLM agents by automatically synthesizing a code harness |
| **机构** | Google DeepMind |
| **作者** | Xinghua Lou, Miguel Lázaro-Gredilla, Antoine Dedieu, Carter Wendelken, Wolfgang Lehrach, Kevin P. Murphy |
| **时间** | 2026年3月5日 |
| **发表** | arXiv:2603.03329v1 |
| **链接** |  |
| **总结** | 本研究旨在解决 LLM 智能体在游戏环境中频繁执行非法动作（illegal moves）的问题。作者提出 AutoHarness 方法，利用 Gemini-2.5-Flash 通过迭代代码精炼（iterative code refinement）自动合成代码约束装置（code harness），借助 Thompson 采样引导的树搜索在程序空间中高效探索。该方法在 145 个 TextArena 游戏上实现了 100% 的合法动作率，使较小的 Gemini-2.5-Flash 模型超越了更大的 Gemini-2.5-Pro；进一步将整个策略编码为纯代码（harness-as-policy），在 16 个单人游戏上取得了高于 Gemini-2.5-Pro 和 GPT-5.2-High 的平均奖励。 |

---

## 1. 研究背景和问题

本研究属于 LLM 智能体（LLM agents）在游戏环境中的规划与决策领域。尽管大语言模型在代码生成和数学推理上取得了显著进展，但作为智能体使用时，它们经常尝试执行环境严格禁止的非法动作——例如在 Kaggle GameArena 国际象棋比赛中，Gemini-2.5-Flash 78% 的失败归因于非法走步而非策略失误。传统方法依赖人工编写约束装置（harness）来过滤非法动作，但这种方式脆弱且无法扩展到每一个新环境。本文的核心问题是：能否让 LLM 自动合成这种代码约束装置，从而在无需人工干预的情况下消除非法动作？

### 1.1 核心假设

本文提出两个层次的假设：（1）LLM 可以通过迭代代码精炼自动生成合法动作验证器（`is_legal_action()`），在各种游戏中实现 100% 合法动作率；（2）更进一步，LLM 可以将整个决策策略编码为纯代码（`propose_action()`），在推理时完全不依赖 LLM 调用，同时获得更优的游戏表现。

---

## 2. 方法

![图1：Code-as-harness 学习流程](2603.03329v1_fig/2603.03329v1-Figure1-1.png)

AutoHarness 将约束装置的生成建模为程序空间上的搜索问题，采用 Thompson 采样引导的树搜索（Tang et al., 2024）来高效探索候选代码。在树结构中，每个节点代表一个代码版本，其启发式值为该代码在环境中的合法动作成功率。搜索过程由三个核心组件驱动：**评估器（Evaluator）** 在 10 个并行环境中执行最多 1000 步的 rollout，检测非法动作和代码执行错误；**评论器（Critic）** 汇总最多 5 个失败步骤的错误信息，提供结构化反馈；**精炼器（Refiner）** 作为变异算子，基于旧代码和错误反馈生成新的候选代码。该方法支持两种约束装置模式：**harness-as-action-filter** 仅学习 `is_legal_action()` 函数用于过滤 LLM 提议的动作（构成拒绝采样器），以及 **harness-as-policy** 同时学习 `propose_action()` 和 `is_legal_action()`，将整个策略编码为纯 Python 代码，在推理时无需调用 LLM。训练使用 Gemini-2.5-Flash，当启发式值达到 1.0 或超时时终止搜索。

---

## 3. 实验

### 3.1 实验设置

| 维度 | 名称 | 数量 |
|---|---|---|
| **模型** | Gemini-2.5-Flash, Gemini-2.5-Pro, GPT-5.2, GPT-5.2-High, Gemini-2.5-Flash+Harness (ours), Harness-as-Policy (ours) | 6 |
| **训练** | TextArena 游戏环境（通过 rollout 产生反馈） | 145 个游戏 |
| **评测** | TextArena 1P 游戏 16 个 + 2P 游戏 16 个（共 32 个游戏） | 32 |
| **指标** | Legal Action Rate, Average Reward (1P), Win/Draw/Loss Rate (2P) | 3 |

### 3.2 实验解析

### 3.2.1 双人游戏对战结果（Figure 3）

![图3：Gemini-2.5-Flash+Harness 对阵 Gemini-2.5-Pro 在 16 个双人游戏上的胜/负/平比率](2603.03329v1_fig/2603.03329v1-Figure3-1.png)

- **图表内容**：该图展示了 AutoHarness 方法（Gemini-2.5-Flash+Harness）与 Gemini-2.5-Pro 在 16 个双人游戏中的胜/负/平比率，横轴为游戏名称，纵轴为比率。
- **揭示关系**：AutoHarness 使较小的 Gemini-2.5-Flash 在 9/16 个游戏中战胜了更大的 Gemini-2.5-Pro，整体胜率达 56.3%（对比 Gemini-2.5-Pro 的 38.2%）。当对阵原版 Gemini-2.5-Flash 时，胜率进一步上升至 64.8%。
- **关键数据**：在 Chess、Othello、Stratego 等复杂策略游戏中，AutoHarness 表现尤为突出，几乎无败绩。

### 3.2.2 单人游戏多智能体对比（Figure 5）

![图5：不同智能体在 16 个 TextArena 单人游戏上的平均奖励](2603.03329v1_fig/2603.03329v1-Figure5-1.png)

- **图表内容**：该图对比了 6 种智能体在 16 个单人游戏上的平均奖励，包括 Gemini-2.5-Flash (0.673)、Gemini-2.5-Pro (0.707)、Gemini-2.5-Flash+Harness (0.745)、GPT-5.2 (0.635)、GPT-5.2-High (0.844) 以及 Harness-as-Policy (0.870)。
- **揭示关系**：Harness-as-Policy 取得了所有智能体中最高的平均奖励 0.870，超越了 GPT-5.2-High (0.844) 和 Gemini-2.5-Pro (0.707)。这表明将策略完全编码为代码不仅消除了 LLM 推理成本，还能获得更优表现。
- **关键数据**：Harness-as-Policy 在单个游戏维度上赢得 3/16 个游戏，GPT-5.2-High 赢 5/16，其余 8/16 打平。由于 Harness-as-Policy 生成纯 Python 代码，其推理成本几乎为零，而 GPT-5.2 系列实验花费约 640 美元。

### 3.2.3 合法动作率分析（Figure 7）

![图7：TextArena 单人游戏逐游戏合法动作成功率](2603.03329v1_fig/2603.03329v1-Figure7-1.png)

- **图表内容**：该表展示了各智能体在 16 个单人游戏上的合法动作成功率（legal action success rate），列出了 Gemini-2.5-Flash、Gemini-2.5-Pro、AutoHarness、GPT-5.2、GPT-5.2-High 和 Harness-as-Policy 的逐游戏数据。
- **揭示关系**：AutoHarness 和 Harness-as-Policy 在所有 16 个游戏上均达到 100% 合法动作率，而其他模型在多个游戏上存在显著的非法动作问题。例如 PegJump-v0 中，Gemini-2.5-Flash 仅 67.97%，GPT-5.2 仅 60.17%。
- **关键数据**：Gemini-2.5-Flash 在 Minesweeper 上仅 88.69%，Gemini-2.5-Pro 在 FifteenPuzzle 上仅 88.14%，凸显了 LLM 在理解复杂游戏规则方面的系统性不足。

---

## 4. 局限性与未来工作

### 4.1 原文描述

作者指出当前方法为每个游戏单独生成一个约束装置，未实现跨游戏的知识迁移。未来工作方向包括：（1）将领域专家（智能体）的知识蒸馏回基础 LLM，实现递归式自我改进；（2）构建可复用的约束装置库；（3）将方法扩展到更具挑战性的多模态游戏环境，如 Craftax 和 Terra Nova。此外，harness-as-policy 目前仅适用于单人游戏，双人游戏需要对对手策略的战略推理，通常需要类 MCTS 方法。

### 4.2 模型总结

本研究的核心贡献在于展示了"用代码约束 LLM"这一范式的有效性，但仍存在若干局限：方法依赖环境提供明确的合法/非法反馈信号，在反馈模糊的真实世界任务中可能难以直接迁移；Thompson 采样树搜索虽然有效，但部分游戏（如 Breakthrough-v0-small）需要 136 次迭代才能收敛，训练效率仍有优化空间；此外，harness-as-policy 生成的纯代码策略在简单规则游戏上表现优异，但对于需要深层战略推理的复杂博弈场景，纯代码策略的上界可能受限。未来可能的研究方向包括将该方法与强化学习或搜索算法结合，探索跨游戏的元学习策略，以及将 code-as-harness 范式推广到机器人控制、对话系统等非游戏领域。（由 Claude Opus 4.6 模型生成）
