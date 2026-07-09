# 每日论文报告 — HarnessX：可组合、自适应、可进化的 Agent Harness 工厂

## 结构化摘要

| 维度 | 内容 |
|---|---|
| **标题** | HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry |
| **机构** | Darwin Agent Team |
| **作者** | Tingyang Chen*, Shuo Lu*, Kang Zhao*, Weicheng Meng, Kun Shao†, Jian Luan†, Hanlin Teng, Tianhao Li, Chao Li, Xule Liu, Jian Liang, Zhizhong Zhang, Yuan Xie, Heng Qu |
| **时间** | 2025年7月 |
| **发表** | ArXiv (2606.14249v2) |
| **链接** | 代码将在未来发布中开源 |
| **总结** | 本研究解决 AI Agent 的运行时 Harness（包括提示词、工具、记忆和控制流）仍然是手工构建且静态不变的问题。HarnessX 将 Harness 视为可组合的一等对象（first-class object），通过 AEGIS——一个基于执行轨迹驱动的多智能体进化引擎——实现自动适应，并通过跨 Harness 版本的 GRPO 训练将 Harness 进化与模型训练闭环。在五个基准测试上，HarnessX 实现了平均 +14.5%（最高 +44.0%）的绝对性能提升，且在弱模型上增益最大，表明 Agent 的进步不必完全依赖模型扩展（model scaling），运行时接口的组合与进化是一条可行的互补路径。 |

---

## 1. 研究背景和问题

AI Agent 的性能不仅取决于底层基础模型，还高度依赖其运行时 Harness——即在模型与环境之间进行中介的提示词模板、工具包装器、记忆策略和控制流。然而当前的 Harness 开发面临三重困境：一是手工构建且静态不变，每次更换模型或任务都需要重新搭建；二是架构上高度耦合，提示词、工具、重试策略和记忆混杂在同一代码路径中，改动一处容易默默破坏其他部分；三是 Harness 工程与模型训练各自为政，执行过程中产生的丰富轨迹数据被白白丢弃。本文的核心问题是：能否将 Harness 视为可组合、可进化的一等对象，使其在执行反馈中持续自我改进，并与模型训练形成闭环优化？

---

## 2. 方法

![图1：AEGIS 进化循环架构。元智能体 M 驱动四个阶段（Digester、Planner、Evolver、Critic），基于确定性门控决定是否发布编辑](2606.14249v2_fig/2606.14249v2-Figure2-1.png)

HarnessX 的方法论由三个层次组成。**Harness 组合层**将 Harness 形式化为一等对象 H = (M, C)，其中 M 为模型配置，C 为 Harness 配置。C 进一步分解为处理器列表 P 和槽资源 S。每个行为由 Processor（处理器）实现，绑定到八个生命周期钩子（Hook）之一，遵循 async def process(event) -> AsyncIterator[Event] 的类型化协议。九维分类法（model selection、context assembly、memory management、tool ecosystem、execution environment、evaluation/reward、control/safety、observability、training bridge）覆盖了完整的行为空间，替换代数（substitution algebra）保证类型安全的组合与替换。**Harness 适应层**引入 AEGIS 引擎，其核心洞察是将 Harness 进化映射为符号空间中的强化学习 MDP——Harness 配置为状态、类型化编辑为动作、执行轨迹与验证器分数为反馈。该映射预测了三种 RL 病理的对应物：奖励黑客（reward hacking）、灾难性遗忘（catastrophic forgetting）和探索不足（under-exploration），分别由 Critic、确定性门控（seesaw constraint）和 Planner 进行防御。AEGIS 的四阶段流水线（Digester 压缩轨迹 → Planner 构建适应格局 → Evolver 生成候选 Harness → Critic 评估 + 确定性门控）确保"LLM 探索和提议，确定性结构决定发布"。**Harness-模型协同进化层**打破了单独进化的两个天花板：Harness-only 的"脚手架天花板"和 Model-only RL 的"训练信号天花板"。通过共享重放缓冲区（shared replay buffer），同一批轨迹同时驱动 AEGIS 的 Harness 更新和跨 Harness GRPO（Cross-Harness GRPO）的模型参数更新——同一任务在不同 Harness 版本下的轨迹被分组，组内相对优势让模型内化最优策略，而 Harness 进化充当模型 RL 的结构化探索算子。

---

## 3. 实验

### 3.1 实验设置

| 维度 | 名称 | 数量 |
|---|---|---|
| **模型** | 元智能体：Claude Opus 4.6；任务智能体：Claude Sonnet 4.6, GPT-5.4, Qwen3.5-9B | 4 |
| **训练** | 协同进化使用 GAIA (103 tasks) 和 WebShop (100 tasks) 的轨迹作为 GRPO 训练数据 | 不适用 |
| **评测** | GAIA (103), ALFWorld (134), WebShop (100), τ3-Bench (3 domains), SWE-bench Verified (55) | 5个基准，共392+任务 |
| **指标** | Pass@2 成功率 (%), 各基准使用特定验证器（精确匹配/目标完成/属性匹配/规则合规/补丁解决） | 1 |

### 3.2 实验解析

### 3.2.1 主实验结果

![图2：主实验结果表，展示 pass@2 成功率（%），Evolved 为达到的最高准确率](2606.14249v2_fig/2606.14249v2-Table4-1.png)

![图3：进化轨迹图，展示 pass@2 成功率随进化轮次的变化，虚线为静态 Harness 基线](2606.14249v2_fig/2606.14249v2-Figure4-1.png)

- **图表内容**：Table 4 和 Figure 4 报告了 15 个模型-基准配置在 Harness 进化前后的 pass@2 成功率。横轴为进化轮次，纵轴为任务成功率。
- **揭示关系**：AEGIS 在 15 个配置中改善了 14 个，平均绝对提升 +14.5%（最高 +44.0%）。存在显著的逆缩放效应（inverse-scaling effect）：基线越弱的任务智能体获益越大（如 Qwen3.5-9B 在 ALFWorld 上从 53.0% 提升至 97.0%，+44.0%），而强模型提升较小（Sonnet 4.6 在同一基准上 +11.2%），表明进化的 Harness 弥补了弱模型无法自我纠正的行为缺口。
- **关键数据**：ALFWorld 增益最大（+11.2% 至 +44.0%）；SWE-bench Verified 三个模型均有 +10.9% 至 +18.2% 的提升；唯一停滞的配置是 GAIA GPT-5.4（Δ=0.0），后通过变体隔离（Ensemble routing）解决，提升至 87.4%。

### 3.2.2 协同进化实验

![图4：协同进化 vs. 仅 Harness 进化对比。星号标记各方法的峰值，阴影区域为协同进化的额外增益](2606.14249v2_fig/2606.14249v2-Figure5-1.png)

- **图表内容**：Figure 5 展示了在 GAIA 和 WebShop 上使用 Qwen3.5-9B 作为任务智能体时，协同进化（交替 Harness 进化与跨 Harness GRPO 模型训练）与仅 Harness 进化的成功率对比。
- **揭示关系**：两条曲线在前 4 轮重合，之后协同进化开始拉开差距并持续保持优势。协同进化打破了仅 Harness 进化的"脚手架天花板"（GAIA ~37%、WebShop ~49%），在 GAIA 上额外提升 +4.3%（37.4% → 41.7%），在 WebShop 上额外提升 +5.0%（49.0% → 54.0%），平均 +4.7%。这证明了 Harness 结构进化与模型参数优化的互补性——进化的 Harness 提供新的行为模式作为探索算子，模型训练则将最优策略内化为参数能力。

---

## 4. 局限性与未来工作

### 4.1 原文描述

论文明确列出五项局限性：(1) **无留出评估集**——所有报告增益均在进化使用的同一任务集上测量，存在选择偏差和潜在过拟合，对未见任务的泛化性未经验证；(2) **仅限离散动作空间**——所有实验使用文本交互，未测试连续动作空间（如机器人控制）；(3) **闭源元智能体**——AEGIS 要求元智能体具备多文件代码生成和结构化轨迹分析能力，开源模型（如 Qwen3.5-72B）作为元智能体的效果未知；(4) **联合控制假设**——协同进化要求同时控制 Harness 进化和模型训练，实际中这通常分属不同团队；(5) **基准覆盖有限**——SWE-bench 仅使用 55 个任务子集，τ3-Bench 仅覆盖三个领域。此外，τ3-Bench Telecom 的失败案例暴露了逐编辑门控（per-edit gating）的结构性局限：亚阈值耦合在积累到超过检测阈值前无法被发现。

### 4.2 模型总结

本文提出了一个富有洞察力的框架，将 Agent Harness 从被动的"脚手架"提升为可组合、可进化的一等对象，但仍有几个方向值得深入探索。首先，泛化性验证是最紧迫的缺口——当前在进化集上报告峰值性能的做法使结论的稳健性存疑，未来需引入留出集和跨分布评估。其次，操作镜像（operational mirror）作为设计启发而非形式化框架，其预测能力的边界有待理论澄清；亚阈值耦合的积累问题暗示需要超越逐编辑门控的全局一致性检测机制。第三，降低对强闭源元智能体的依赖（例如探索开源替代或层次化的元智能体架构）将大幅扩展该方法的可及性。最后，将该框架推广至多模态和连续控制场景，以及探索在线部署中的持续进化而非离线批量进化，可能是有价值的新研究方向。（由 Claude Opus 4.6 模型生成）
