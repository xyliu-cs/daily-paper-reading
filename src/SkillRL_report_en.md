# Daily Paper Reading — SKILLRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning

## Structured Summary

| Dimension | Content |
|---|---|
| **Title** | SKILLRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning |
| **Institution** | UNC-Chapel Hill, University of Chicago, University of California San Diego |
| **Authors** | Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, Zeyu Zheng, Cihang Xie, Huaxiu Yao |
| **Date** | February 10, 2026 |
| **Venue** | arXiv preprint arXiv:2602.08234v1 |
| **Link** | https://github.com/aiming-lab/SkillRL |
| **Summary** | This paper addresses the inability of LLM agents to accumulate reusable decision knowledge from past interactions. It proposes SKILLRL, which distills raw trajectories into a hierarchical skill library (SKILLBANK) and co-evolves that library with the agent's RL policy through recursive failure analysis. SKILLRL achieves 89.9% on ALFWorld and 72.7% on WebShop, outperforming GRPO by 12.3% and 6.6% respectively, and surpassing GPT-4o and Gemini-2.5-Pro by over 29 percentage points on ALFWorld. |

---

## 1. Background and Problem

This work sits at the intersection of LLM agent systems and reinforcement learning, tackling a fundamental limitation: current LLM agents treat every task as an isolated episode, unable to learn from past successes or failures, which significantly hinders their evolution in complex long-horizon tasks. Existing memory-based approaches store raw trajectories directly, but these are often "lengthy and contain significant redundancy and noise," making it difficult to extract critical decision principles (Chhikara et al., 2025). The core question motivating this paper is: *how can agents efficiently learn from experience and transfer that knowledge to other tasks?*

### 1.1 Core Hypothesis

The paper's central hypothesis is that effective experience transfer requires **abstraction** rather than memorization — analogous to how human experts develop compact, reusable skills rather than recalling every past action verbatim. A static skill library is insufficient; the library must dynamically co-evolve with the agent's improving policy to cover newly discovered failure modes that emerge as the agent explores new state regions.

---

## 2. Method

![Figure 2: Overview of the SKILLRL framework. Trajectories are collected with a base model, distilled into a hierarchical skill library, followed by cold-start SFT, and then RL training with dynamic skill evolution based on validation failures.](2602.08234v1_fig/2602.08234v1-Figure2-1.png)

SKILLRL comprises three tightly integrated components. **Experience-based Skill Distillation**: A base policy $\pi_{\text{base}}$ collects successful trajectories $\mathcal{T}^+$ and failed trajectories $\mathcal{T}^-$ through environment rollouts. A teacher model $\mathcal{M}_T$ applies differential processing — extracting generalizable strategic patterns $s^+ = \mathcal{M}_T(\tau^+, d)$ from successes and synthesizing counterfactual failure lessons $s^- = \mathcal{M}_T(\tau^-, d)$ from failures, achieving 10–20× token compression relative to raw trajectories. **Hierarchical Skill Library (SKILLBANK)**: Skills are organized into General Skills $\mathcal{S}_g$ (universal strategic principles: systematic search, state management, goal-tracking) and Task-Specific Skills $\mathcal{S}_k$ (domain-specific action sequences and failure mitigations per task category $k$). At inference, task-specific skills are retrieved via semantic similarity $\mathcal{S}_{\text{ret}} = \text{TopK}(\{s \in \mathcal{S}_k : \text{sim}(e_d, e_s) > \delta\}, K)$. A cold-start SFT phase precedes RL to teach the base model how to interpret and apply retrieved skills. **Recursive Skill Evolution**: After each validation epoch, categories where $\text{Acc}(C) < \delta$ trigger collection of failed validation trajectories $\mathcal{T}_{\text{val}}^-$. The teacher model analyzes these failures against the current SKILLBANK to identify gaps and generate refined or new skills $\mathcal{S}_{\text{new}} = \mathcal{M}_T(\mathcal{T}_{\text{val}}^-, \text{SKILLBANK})$, updating the library as $\text{SKILLBANK} \leftarrow \text{SKILLBANK} \cup \mathcal{S}_{\text{new}}$. Policy optimization uses GRPO on skill-augmented trajectories with a KL penalty anchored to $\pi_{\text{ref}} = \pi_{\theta_\text{sft}}$ to preserve skill utilization capabilities.

---

## 3. Experiments

### 3.1 Experimental Setup

| Dimension | Name | Quantity |
|---|---|---|
| **Models** | Qwen2.5-7B-Instruct (base model), OpenAI o3 (teacher model) | 2 |
| **Training** | ALFWorld interaction trajectories, NQ, HotpotQA | Not explicitly stated |
| **Evaluation** | ALFWorld (6 subtasks), WebShop, NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle | 9 benchmarks |
| **Metrics** | Success Rate (%), Average Score | 2 |

Training configuration: GRPO with learning rate $1 \times 10^{-6}$, batch size 16, group size 8, 4 gradient accumulation steps; retrieval count $K=6$, failure collection threshold $\delta=0.4$.

### 3.2 Experimental Analysis

#### 3.2.1 Main Results: ALFWorld and WebShop

Table 1 (paper page 6) compares SKILLRL against four categories of baselines: closed-source LLMs, prompt-based/memory-based methods, RL-based methods, and memory-augmented RL methods, reporting per-subtask and overall success rate (%) on ALFWorld and both score and success rate on WebShop.

- **Figure/Table Content**: ALFWorld subtasks include Pick, Look, Clean, Heat, Cool, and Pick2; the table reports individual and overall success rates alongside WebShop score and success rate.
- **Revealed Insights**: SKILLRL achieves 89.9% overall on ALFWorld and 72.7% on WebShop, consistently outperforming all baselines across every subtask, with the largest gains on challenging multi-step planning tasks (Cool: +22% over GRPO, Pick2: +23%), confirming that structured skill priors effectively accelerate policy learning in sparse-reward settings.
- **Key Data**: The 12.3% absolute improvement over GRPO (from 77.6% to 89.9%) is directly attributable to skill augmentation rather than algorithmic variance. SKILLRL with a 7B model surpasses GPT-4o by 41.9% and Gemini-2.5-Pro by 29.6% on ALFWorld, demonstrating that effective skill learning can compensate for model scale.

#### 3.2.2 Generalization on Search-Augmented QA Tasks

![Table 2: Performance on search-augmented QA tasks. SKILLRL is trained on NQ and HotpotQA and evaluated on 7 benchmarks including in-domain and out-of-domain datasets.](2602.08234v1_fig/2602.08234v1-Table2-1.png)

- **Figure/Table Content**: Table 2 reports accuracy (%) on single-hop QA (NQ, TriviaQA, PopQA) and multi-hop QA (HotpotQA, 2Wiki, MuSiQue, Bamboogle), with an overall average column.
- **Revealed Insights**: SKILLRL achieves a 47.1% average, outperforming EvolveR (43.1%) and Search-R1 (38.5%); strong performance on OOD datasets like TriviaQA and 2Wiki confirms that distilled search strategies transfer as task-agnostic heuristics.
- **Key Data**: On Bamboogle (the hardest multi-hop task), SKILLRL reaches 73.8% versus EvolveR's 54.4%, a 19.4-point gap demonstrating hierarchical skills' advantage for multi-step information synthesis.

#### 3.2.3 Ablation Study and Skill Library Evolution

![Table 3: Ablation study results. Average success rate (%) on ALFWorld and WebShop across four ablation dimensions.](2602.08234v1_fig/2602.08234v1-Table3-1.png)

![Figure 3: Evolution of skill library size during RL training. Dynamic skill evolution adds skills at validation checkpoints, growing from 55 to 100 skills by step 150.](2602.08234v1_fig/2602.08234v1-Figure3-1.png)

- **Figure/Table Content (Table 3)**: Four ablations are evaluated — removing hierarchical structure (task-specific skills only), replacing the skill library with raw trajectories, removing cold-start SFT, and removing dynamic evolution — measuring impact on ALFWorld and WebShop success rates.
- **Revealed Insights**: Replacing skills with raw trajectories causes the largest degradation (~28.2% on ALFWorld), directly validating the abstraction hypothesis; the absence of cold-start SFT causes a 24.7% drop, confirming the base model requires explicit demonstration of skill usage before RL; dynamic evolution contributes a 5.5% gain from the co-evolving library.
- **Figure/Table Content (Figure 3)**: Task-specific skills grow from 43 to 80 (dominant driver), while general skills grow from 12 to 20, with additions concentrated at validation checkpoints where failure patterns trigger targeted skill generation — demonstrating the virtuous cycle between policy improvement and knowledge expansion.

---

## 4. Limitations and Future Work

### 4.1 Original Description

The paper does not include a dedicated limitations section. The conclusion focuses on the positive contributions: "By distilling raw trajectories into compact, reusable skills and enabling dynamic skill evolution during training, SKILLRL achieves state-of-the-art performance on ALFWorld and WebShop while using substantially less context than memory-based approaches." No explicit failure cases or boundary conditions are discussed in the main text.

### 4.2 Model Summary

SKILLRL has several practical limitations worth noting. First, the skill distillation quality is bottlenecked by the teacher model (OpenAI o3 in this work); replacing it with a weaker or locally-hosted model may degrade skill quality, limiting reproducibility in resource-constrained settings. Second, evaluation is confined to two relatively structured environments (ALFWorld, WebShop) and text-based search tasks — scalability to truly open-ended domains (e.g., code execution, real-world web agents with adversarial content) remains unverified. Third, the SKILLBANK lacks explicit mechanisms for skill pruning, merging, or forgetting, which could lead to knowledge redundancy or conflicting skills as the library scales. Future directions could include self-distillation with smaller models to reduce teacher dependency, cross-domain skill transfer to test the universality of extracted principles, and formal skill quality evaluation methods that can detect when skills become stale or contradictory. (Generated by Claude Sonnet 4.6)
