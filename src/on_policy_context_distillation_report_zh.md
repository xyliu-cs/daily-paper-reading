# 每日论文报告 — On-Policy Context Distillation for Language Models

## 结构化摘要

| 维度 | 内容 |
|---|---|
| **标题** | On-Policy Context Distillation for Language Models |
| **机构** | Microsoft Research |
| **作者** | Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, Furu Wei |
| **时间** | 2026年2月12日 |
| **发表** | arXiv:2602.12275v1 |
| **链接** | https://aka.ms/GeneralAI |
| **总结** | 现有上下文蒸馏（Context Distillation）方法依赖离策略（off-policy）训练和前向KL散度最小化，存在暴露偏差和模式覆盖问题。本文提出在策略上下文蒸馏（On-Policy Context Distillation，OPCD），让学生模型在自身生成的轨迹上训练，同时最小化相对于上下文条件教师模型的反向KL散度。OPCD在数学推理、文字游戏和领域特定任务上持续优于基线，且能有效缓解遗忘并支持跨规模蒸馏。 |

---

## 1. 研究背景和问题

本研究属于大语言模型（Large Language Model，LLM）的知识内化与模型蒸馏领域。大型语言模型的上下文学习能力虽强，但上下文中的知识是临时性的——一旦上下文重置，模型就需要重新从提示词中学习。已有的上下文蒸馏方法（Context Distillation）试图将上下文知识压缩进模型参数，但均依赖离策略的前向KL最小化，导致暴露偏差（exposure bias）以及概率质量分散到低质量输出的"幻觉"问题。本文核心问题是：**如何将临时的上下文知识高效、稳健地内化为模型的永久参数？**

---

## 2. 方法

![图1：OPCD总体框架](on_policy_context_distillation_fig/on_policy_context_distillation-Figure1-1.png)

OPCD的核心思想是将在策略蒸馏（on-policy distillation）与上下文蒸馏（context distillation）相结合。在每个训练步骤中，学生模型 $\pi_\theta$ 在**不提供上下文 $c$** 的条件下对输入 $x$ 自由采样生成响应 $y$，随后计算学生分布与上下文条件教师分布 $\pi_\text{teacher}(\cdot \mid c, x, y_{<t})$ 之间的逐令牌（token-level）**反向KL散度**并反向传播。损失函数定义为：

$$\mathcal{L}(\theta) = \mathbb{E}_{(x,c)\sim\mathcal{D},\, y\sim\pi_\theta(\cdot|x)}\left[\frac{1}{|y|}\sum_{t=1}^{|y|} D_\text{KL}\!\left(\pi_\theta(\cdot \mid x, y_{<t}) \,\|\, \pi_\text{teacher}(\cdot \mid c, x, y_{<t})\right)\right]$$

反向KL鼓励**模式寻找（mode-seeking）**行为，使学生专注于教师分布的高概率区域，避免了前向KL导致的模式覆盖。实现中，KL散度通过仅对学生模型预测概率最高的 top-$k$ 个令牌求和来近似计算。框架支持两种教师配置：**教师-学生蒸馏**（$\pi_\text{teacher} \neq \pi_\theta$，默认配置，教师参数冻结或定期更新）和**自蒸馏**（$\pi_\text{teacher} = \pi_\theta$，教师与学生共享权重）。论文验证了OPCD在两类应用上的有效性：一是**经验性知识蒸馏**（Experiential Knowledge Distillation），模型从历史解题轨迹中提炼可迁移知识并内化；二是**系统提示词蒸馏**（System Prompt Distillation），将优化后的系统提示词编码的行为内化到模型参数中。

---

## 3. 实验

### 3.1 实验设置

| 维度 | 名称 | 数量 |
|---|---|---|
| **模型** | Qwen3-8B, Qwen3-4B, Qwen3-1.7B, Qwen3-4B-Instruct-2507, Qwen2.5-7B-Instruct, Qwen2.5-3B-Instruct, Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct | 8 |
| **训练** | DAPO-Math-17K（数学）、TextArena Frozen Lake & Sokoban（文字游戏）、MedMCQA + 安全数据集（系统提示词蒸馏） | Math ~14K样本；文字游戏多轮交互；系统提示词任务各训练集 |
| **评测** | 数学测试集（1000样本）、文字游戏测试集（128样本）、MedMCQA（500样本）、安全分类（500样本）、IF-Eval（OOD）| 各任务分别评测 |
| **指标** | 任务准确率（Accuracy）、IF-Eval严格准确率（OOD评测） | 2 |

### 3.2 实验解析

#### 3.2.1 经验性知识蒸馏：测试时场景 vs 过滤场景

![表1：测试时经验性知识整合结果](on_policy_context_distillation_fig/on_policy_context_distillation-Table1-1.png)

- **图表内容**：表1对比了在测试时经验性知识整合场景下，基础模型、In-Context、Context Distillation和OPCD四种方法在数学（Qwen3-8B）和Frozen Lake游戏（Qwen3-1.7B）上的任务准确率及OOD IF-Eval得分。
- **揭示关系**：OPCD在两类任务上均超过了离策略上下文蒸馏基线，同时在OOD指标上也优于或持平于基础模型，表明OPCD在内化上下文知识的同时有效缓解了灾难性遗忘。
- **关键数据**：Qwen3-8B数学任务：基础模型75.0 → Context Distill. 78.5 → OPCD **79.7**，IF-Eval：81.3 → 81.2 → **81.7**；Qwen3-1.7B Frozen Lake：基础模型6.3 → Context Distill. 22.9 → OPCD **26.5**。

#### 3.2.2 系统提示词蒸馏：分布内与分布外的权衡

![图3：OPCD与离策略上下文蒸馏在安全系统提示词蒸馏任务上的对比](on_policy_context_distillation_fig/on_policy_context_distillation-Figure3-1.png)

- **图表内容**：图3展示了在安全系统提示词蒸馏过程中，两种方法随训练步数变化时的分布内（安全任务）准确率（左）和分布外（医疗任务）准确率（右）。
- **揭示关系**：OPCD在分布内性能上始终高于离策略基线，同时在OOD任务上超过离策略基线约4个百分点，验证了在策略采样能有效缓解遗忘问题，而离策略方法在OOD性能上出现明显下降。

#### 3.2.3 跨规模蒸馏

![图2：OPCD在不同规模学生模型上的经验性知识蒸馏结果](on_policy_context_distillation_fig/on_policy_context_distillation-Figure2-1.png)

- **图表内容**：图2展示了使用冻结的Qwen3-8B教师生成的经验性知识，分别对Qwen3-1.7B、Qwen3-4B、Qwen3-8B三种规模的学生模型进行OPCD训练后的测试准确率，并与In-Context直接注入和初始学生模型进行对比。
- **揭示关系**：OPCD在所有规模的学生模型上均持续提升性能，而直接将教师生成的经验性知识注入更小模型的上下文反而会降低性能，这说明经验性知识与消费该知识的模型之间的在策略对齐至关重要。

---

## 4. 局限性与未来工作

### 4.1 原文描述

论文结论部分指出："Our work opens avenues for future research on continual accumulation of experiential knowledge, adaptive context selection strategies, and scaling OPCD to broader domains requiring persistent knowledge internalization."（"我们的工作为未来研究开辟了道路，包括经验性知识的持续积累、自适应上下文选择策略，以及将OPCD扩展到需要持久知识内化的更广泛领域。"）

### 4.2 模型总结

OPCD在设计上依赖白盒访问教师模型的令牌级概率，这限制了其直接应用于黑盒API场景（尽管有工作正在探索黑盒扩展）。此外，反向KL散度虽然缓解了模式覆盖，但在教师分布极度多峰时可能导致模式崩塌；经验性知识提取质量对最终蒸馏效果有较大影响，如何自动评估和筛选高质量经验知识仍是开放问题。未来工作可探索：在线动态更新经验池的持续学习框架、将OPCD与强化学习与可验证奖励（RLVR）结合以实现更强的推理内化，以及多智能体场景下的分布式经验积累与蒸馏机制。（由 Claude Sonnet 4.6 模型生成）
