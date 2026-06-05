# Daily Paper Reading — Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents

## Structured Summary

| Dimension | Content |
|---|---|
| **Title** | Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents |
| **Institution** | The Pennsylvania State University, UC Santa Cruz, Amazon |
| **Authors** | Minhua Lin, Juncheng Wu, Zijun Wang, Zhan Shi, Yisi Sang, Bing He, Zewen Liu, Tianxin Wei, Zongyu Wu, Zhiwei Zhang, Dakuo Wang, Xiang Zhang, Benoit Dumoulin, Cihang Xie, Yuyin Zhou, Suhang Wang, Hanqing Lu |
| **Date** | May 2025 |
| **Venue** | Arxiv |
| **Link** |  |
| **Summary** | This research addresses the unclear relationship between a model's base task-solving capability and its capabilities in harness self-evolution for LLM agents. The authors decompose harness self-evolution into two independent dimensions — harness-updating (producing useful harness updates) and harness-benefit (benefiting from updated harnesses during task solving) — and conduct systematic cross-pairing experiments across seven LLMs and three agentic benchmarks. The key findings are that harness-updating is flat across capability tiers (even Qwen3.5-9B matches Claude Opus 4.6), while harness-benefit is non-monotonic: mid-tier models benefit most, and weak-tier models benefit least due to failures in harness activation and adherence. |

---

## 1. Background and Problem

LLM agents are increasingly deployed as systems built around editable external harnesses — prompts, skills, memories, and tools — that shape task execution without changing model parameters. Harness self-evolution adapts these agents by updating harness components from execution evidence, but existing evaluations only report end-to-end performance gains that conflate the evolver's contribution with the task-solving agent's ability to leverage updated harnesses. This paper asks two core questions: which models produce useful harness updates, and which models benefit most from them?

### 1.1 Core Hypothesis

The authors hypothesize that a model's base task-solving capability systematically decouples from its two harness-evolution capabilities: a strong base capability does not necessarily predict better harness-updating ability or greater harness-benefit, and these two evolution capabilities follow distinct patterns across the capability spectrum.

---

## 2. Method

![Figure 1: Overview of findings](2605.30621v1_fig/2605.30621v1-Figure2-1.png)

The paper formalizes a harness self-evolution protocol and capability measurement framework. An agent is defined as A = (f, H), combining a frozen model backbone f with a harness state H. The evolver produces incremental harness updates from execution evidence D. Three metrics are defined: (1) base capability M_base(f), measuring task-solving performance under the initial harness; (2) harness-updating gain Δ_update(e), the mean pairwise gain across anchor agents when varying the evolver; and (3) harness-benefit gain Δ_benefit(f), the maximum gain across anchor evolvers when varying the task-solving agent. The experimental design uses a cross-pairing approach: seven LLMs serve as both agents and evolvers across three benchmarks (SWE-bench Verified, MCP-Atlas, SkillsBench), sharing identical initial harnesses, task streams, and prompt templates with only the LLM backbone varying, thereby enabling independent measurement of both capabilities.

---

## 3. Experiments

### 3.1 Experimental Setup

| Dimension | Name | Quantity |
|---|---|---|
| **Models** | Claude Opus 4.6, Claude Sonnet 4.6, Claude Haiku 4.5, Qwen3-235B-A22B, Qwen3-32B, GPT-OSS-120B, Qwen3.5-9B | 7 |
| **Training** | Not applicable (harness self-evolution does not update model weights) | N/A |
| **Evaluation** | SWE-bench Verified (SWE), MCP-Atlas (MCP), SkillsBench (SB) | 3 |
| **Metrics** | Pass Rate, Harness-updating Gain (Δ_update), Harness-benefit Gain (Δ_benefit), Skill-Load Rate (SLR), Harness-Following Rate (HFR) | 5 |

### 3.2 Experimental Analysis

### 3.2.1 Evolver-side Analysis: Harness-updating Is Flat

![Figure 2: Harness-updating capability across evolvers](2605.30621v1_fig/2605.30621v1-Figure3-1.png)

- **Figure/Table Content**: Figure 3 presents the harness-updating capability Δ_update (in percentage points, pp) of seven models serving as evolvers across three benchmarks, grouped by model family (Claude, Qwen, GPT-OSS).
- **Revealed Insights**: Harness-updating is flat across base-capability tiers — the gap between the best and worst evolver is at most 3.1 pp on any benchmark, and no single evolver dominates across all benchmarks.
- **Key Data**: The smallest model, Qwen3.5-9B, achieves the highest update gain on SkillsBench (3.8 pp), exceeding both Claude Opus 4.6 (2.3 pp) and Qwen3-235B (1.5 pp). Qwen3-235B leads on SWE (8.2 pp) but ranks last on MCP (0.6 pp).

### 3.2.2 Agent-side Analysis: Harness-benefit Is Non-monotonic

![Figure 3: Base pass rate and harness-benefit across benchmarks](2605.30621v1_fig/2605.30621v1-Table1-1.png)

- **Figure/Table Content**: Table 1 reports the base pass rate (%) and harness-benefit gain Δ_benefit (pp) for six LLM backbones serving as task-solving agents, across all three benchmarks.
- **Revealed Insights**: Harness-benefit does not increase monotonically with base capability. Mid-tier models benefit the most (e.g., Qwen3-235B gains 19.3 pp on SWE; GPT-OSS-120B gains 7.0 pp on MCP), while both weak-tier (Qwen3-32B) and strong-tier (Opus 4.6) models gain less. The low gains at the strong end are explained by a ceiling effect, while the low gains at the weak end are traced to two distinct failure modes.
- **Key Data**: Opus 4.6 starts at 74.2% base pass rate on SWE with only 2.6 pp gain, whereas Qwen3-235B starts at 20.7% and gains 19.3 pp.

### 3.2.3 Diagnosing Weak-tier Failure Modes

![Figure 4: Two harness-benefit failure modes on SkillsBench](2605.30621v1_fig/2605.30621v1-Figure7-1.png)

- **Figure/Table Content**: Figure 7 illustrates two failure modes through concrete SkillsBench examples. The left panel shows harness activation failure, where Qwen3-32B embeds the skill-loading request inside a multi-key action rather than issuing a standalone command. The right panel shows harness adherence failure, where the model loads the skill but treats it as a literal script rather than procedural guidance.
- **Revealed Insights**: Weak-tier models exhibit dramatically lower skill-load rates (SLR: 0.251 for Qwen3-32B vs. 0.957 for Opus 4.6) and lower harness-following rates even when loaded (HFR: 0.142 vs. 0.757). Critically, adherence degrades over the trajectory — Qwen3-32B's score drops from 0.52 after harness loading to 0.13 at final validation, while Opus 4.6 remains stable from 0.89 to 0.80, revealing a long-horizon instruction-following bottleneck.

---

## 4. Limitations and Future Work

### 4.1 Original Description

The authors acknowledge several limitations: (1) the study focuses exclusively on harness self-evolution with fixed model weights, excluding parametric fine-tuning, reinforcement learning, or hybrid adaptation methods; (2) the model set is representative but not exhaustive — a broader model grid would further clarify how harness-updating and harness-benefit vary with model family, scale, training recipe, and deployment cost; (3) harness self-evolution raises deployment concerns, as incorrect lessons, unsafe tool-use rules, biased instructions, or sensitive information could persist in the harness and be reused by future agents.

### 4.2 Model Summary

This study's central contribution lies in revealing the systematic decoupling between harness-updating and harness-benefit capabilities, providing clear resource allocation guidance for agentic system design. However, the experimental design relies on fixed prompt templates and a single evolution protocol, which may not fully capture the effects of alternative evolution strategies (e.g., code-level harness optimization). The diagnosis of weak-tier failure modes — activation failure and adherence failure — is thorough, but how to effectively address these through agent training (particularly long-horizon instruction following) remains an open question. Promising future directions include: treating harness invocation as a first-class trained skill, designing training objectives that target instruction-following decay over long trajectories, and exploring synergies between parametric updates and harness evolution. (Generated by Claude Opus 4.6 model)
