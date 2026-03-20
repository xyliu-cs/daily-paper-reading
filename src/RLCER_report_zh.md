# 每日论文报告 — Reinforcing Chain-of-Thought Reasoning with Self-Evolving Rubrics

## 结构化摘要

| 维度 | 内容 |
|---|---|
| **标题** | Reinforcing Chain-of-Thought Reasoning with Self-Evolving Rubrics |
| **机构** | ByteDance Seed、新加坡国立大学（NUS）、中国科学技术大学（USTC） |
| **作者** | Leheng Sheng, Wenchang Ma, Ruixin Hong, Xiang Wang, An Zhang, Tat-Seng Chua |
| **时间** | 2026年2月12日 |
| **发表** | arXiv 预印本 |
| **链接** | https://alphalab-ustc.github.io/rlcer-alphalab/ |
| **总结** | 该研究旨在解决 RLVR（带可验证奖励的强化学习）训练中缺乏对思维链（CoT）直接监督信号的问题。提出 RLCER 方法，让策略模型同时扮演"推理者"和"评分标准生成者（Rubricator）"两个角色，通过自提议、自进化的评分标准（Rubric）对 CoT 质量进行奖励。实验表明 RLCER 在数学推理和通用知识推理基准上均优于基线 RLVR，且生成的 Rubric 可作为推理提示进一步提升推理性能。 |

---

## 1. 研究背景和问题

本文属于大语言模型（LLM）推理能力增强领域，聚焦于基于强化学习的思维链优化。现有的 RLVR（Reinforcement Learning with Verifiable Rewards）范式仅奖励最终答案的正确性，忽略了对 CoT 本身的直接监督，导致模型容易收敛于次优推理策略，如走捷径或依赖脆弱模式 [7, 16]。训练额外奖励模型（Reward Model）需要大量精细标注，且静态奖励模型无法适应训练中不断变化的 CoT 分布，这两大挑战共同阻碍了实践中 CoT 监督的落地。

### 1.1 核心假设

作者的核心研究问题是："策略模型能否自主提议 CoT 监督标准并在训练中不断自我进化，从而在无任何人工标注的条件下实现对 CoT 的有效奖励？"（"Can the policy model self-propose rubrics as CoT supervision criteria and self-evolve them during training with no human annotations?" [原文 §1]）。若成立，则意味着一种全新的自我改进推理范式，将强化学习的优化目标从"答什么"拓展至"如何思考"。

---

## 2. 方法

![图1：RLCER 核心思路概览——策略模型 πθ 同时扮演推理者和评分标准生成者两个角色](RLCER_fig/RLCER-Figure2-1.png)

RLCER 的核心是让单一策略模型 $\pi_\theta$ 在不同提示下扮演两个角色：**推理者（Reasoner）** $\pi_\theta^{Rea}$ 负责给定问题 Q 生成 CoT $\hat{C}$ 和最终答案 $\hat{A}$；**评分标准生成者（Rubricator）** $\pi_\theta^{Rub}$ 基于问题和采样到的 CoT，生成 K 条文本形式的评分标准（Rubric），每条标准包含文本准则 $\hat{c}_k$（如"避免区间验证后的切题偏移"）和重要性分值 $\hat{s}_k$。一个独立的冻结验证器 $\pi_\phi$ 逐条判断 CoT 是否满足各 rubric，满足度向量 $v_k$ 与最终答案正确率向量 $z$ 的相关性超过阈值（$\text{corr}(v_k, z) > \alpha$，默认 $\alpha=0.2$）且具有判别性（$\text{std}(v_k) > 0$）的 rubric 被认定为"有效"，只有有效 rubric 才参与 CoT 奖励的加权求和与归一化（公式 6）。推理者的总奖励为结果奖励（正确 +1，错误 -1）与归一化 CoT 奖励之和；评分标准生成者则通过有效 rubric 比例 $r^{Rub}_{evolving} = K_{valid}/K$ 作为进化奖励，驱动其持续提议更具挑战性、更具信息量的评分标准，从而防止 rubric 饱和。两个角色的优势函数分别计算，梯度汇总后共同更新同一组参数 $\theta$，采用基于 DAPO 的 PPO 目标函数（公式 13）实现端对端联合优化。

![图2：RLCER 奖励计算流程——推理者生成 N 条响应，评分标准生成者针对每条 CoT 提议 K 条 Rubric，有效 Rubric 比例驱动自进化](RLCER_fig/RLCER-Figure4-1.png)

---

## 3. 实验

### 3.1 实验设置

| 维度 | 名称 | 数量 |
|---|---|---|
| **模型** | Qwen3-8B-Base、Qwen3-4B-Base（冷启动 SFT 后进行 RL 训练） | 2 |
| **训练** | DAPO-Math-17k 数学问题数据集 | 17,000 条 |
| **评测** | AIME2024、AIME2025、AMC2023（数学推理）；GPQA-Diamond、SuperGPQA-Eng、SuperGPQA-Med、SuperGPQA-Sci（通用知识推理） | 7 个基准（每个 SuperGPQA 子集 100 题） |
| **指标** | pass@1（每题采样 16 次响应的平均通过率，温度 0.7） | 1 |

### 3.2 实验解析

### 3.2.1 RQ1：自提议 Rubric 是否能提供可靠的强化学习信号？

![图3：仅使用自提议 Rubric 奖励 CoT 时（无结果奖励）的准确率动态](RLCER_fig/RLCER-Figure5-1.png)

- **图表内容**：对比在仅用自提议 rubric 奖励 CoT（无结果奖励，Rubric Only）与使用 0-1 随机值作为 rubric 奖励（Rubric Only Random）两种设置下，训练过程中 AMC23 和 AIME25 准确率的滚动均值曲线。
- **揭示关系**：仅用自提议 rubric 奖励 CoT 仍能带来持续且稳定的性能提升，而随机 rubric 在约 200 步附近出现性能骤降，说明自提议 rubric 产生的奖励信号是有意义的，而非噪声，为 RLCER 方法的可行性奠定了实验基础。

### 3.2.2 RQ2：RLCER 与 RLVR 基线的整体性能对比

![表1：在多个推理基准上的性能对比（pass@1，%）](RLCER_fig/RLCER-Table1-1.png)

- **图表内容**：在 4B 和 8B 两种模型规模下，对比 Base、SFT、+RLVR 和 +RLCER 四个阶段在 7 个推理基准上的 pass@1 准确率。
- **揭示关系**：RLCER 在大多数基准上优于 RLVR，8B 模型的提升尤为显著（AIME2024：34.79% → 37.50%；GPQA-Diamond：46.56% → 48.77%），且尽管只在数学数据集上训练，RLCER 对通用知识推理任务也有良好的泛化能力，体现了自提议 rubric 带来的"免费午餐"效果。

### 3.2.3 RQ3：Rubric 自进化机制的内部分析

![图4：自进化奖励对 Rubric 相关性（左）和 CoT 奖励动态（右）的影响](RLCER_fig/RLCER-Figure6-1.png)

- **图表内容**：训练过程中，rubric 满足度与最终答案正确率的平均相关性 $\text{corr}(v_k, z)$（左图）以及推理者所获 CoT 奖励 $r^{Rea}_{cot}$（右图）随训练步数的变化，对比开启/关闭自进化奖励两种设置。
- **揭示关系**：开启自进化奖励后相关性持续上升，说明模型逐渐学会提出与最终正确率更契合的高质量 rubric；同时 CoT 奖励呈下降趋势，表明进化后的 rubric 难度不断提升，持续为推理者施加有效压力，有效防止了 rubric 饱和问题。

---

## 4. 局限性与未来工作

### 4.1 原文描述

"On the one hand, the introduction of the rubricator role increases the rollout burden and thereby requires more training time. On the other hand, our method is still quite limited to the RLVR domain, leaving the effectiveness of rewarding with self-proposed rubrics on non-verifiable domains unknown." [原文 §6]

作者在结论中提及未来将探索 RLCER 向非可验证域的泛化。

### 4.2 模型总结

RLCER 在有可验证结果的任务（如数学）上效果显著，但在开放生成、对话对齐等非可验证场景中的适用性仍未经验证，这是最根本的局限。评分标准生成者的引入实际上使每步 rollout 计算量成倍增加（需对每条 CoT 再采样 N 次用于计算相关性），对计算资源要求较高，制约了其在超大规模模型上的可扩展性。此外，当前验证器 $\pi_\phi$ 是独立微调并冻结的，其质量直接影响 rubric 有效性的判断，如何使验证器也参与自进化是一个有趣的开放问题。未来方向包括：将自进化 rubric 机制迁移至非可验证域（如用偏好模型或人类反馈替代二元验证器）、降低 rubricator 的计算开销、探索 rubric 在跨任务场景下的迁移能力，以及研究更长训练周期下 rubric 自进化的稳定性。（由 Claude Sonnet 4.6 模型生成）
