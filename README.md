<div align="center">
  <img src="assets/Inspector_Banana.png" alt="Inspector Banana - Slug Search Project" width="200"/>
</div>

# Slug-Search: Agentic RAG for Multi-Hop Question Answering

Slug-Search is a research project focused on enhancing Large Language Models (LLMs) for complex, knowledge-intensive tasks using an "Agentic RAG" (Retrieval Augmented Generation) approach. This project investigates how LLMs can learn to effectively utilize a retrieval system as an interactive tool, guided by reinforcement learning, particularly in the context of multi-hop question answering.

This project builds upon the framework of [ART (Agent Reinforcement Trainer)](./ART_README.md) to explore and develop these agentic systems.

## Core Idea

The central concept of Slug-Search is to empower LLMs with dynamic retrieval capabilities. Instead of a static, one-off retrieval step, we formulate the retrieval system as a tool that the LLM can call multiple times within a single query. This allows the agent to iteratively gather and synthesize information from various sources, a crucial skill for tackling multi-hop questions that require reasoning over several pieces of evidence.

Our methodology focuses on fine-tuning the "Reader" LLM component. This is achieved using outcome-based reinforcement learning (e.g., GRPO), where the model learns from the success of its interactions and tool use, without needing to jointly train the underlying retrieval system.

## Inspiration

Our work is inspired by a wave of recent advancements in agentic search, reasoning, and tool integration for LLMs:

*   Search-o1: Agentic Search-Enhanced Large Reasoning Models (Jan 2025)
*   R1-Searcher: Incentivizing the Search Capability in LLMs via RL (Mar 2025)
*   ReSearch: Learning to Reason with Search for LLMs via RL (Mar 2025)
*   Search-R1: Training LLMs to Reason and Leverage Search Engines w/ RL (Mar 2025)
*   ARTIST: Agentic Reasoning and Tool Integration for LLMs via RL (Apr 2025)

## Project Goals

The primary objectives of the Slug-Search project are:

*   To implement and refine an Agentic-RAG system tailored for multi-hop question-answering scenarios.
*   To fine-tune an LLM (acting as the "Reader") using outcome-based reinforcement learning, enhancing its ability to strategically deploy a search/retrieval tool.
*   To rigorously evaluate the system's performance against relevant baselines, aiming for demonstrable improvements in task success rates and reasoning capabilities.
*   To explore and analyze various design choices concerning language models, retrieval mechanisms, prompting strategies, and reinforcement learning algorithms.

## Key Components & Design Considerations

The project involves careful consideration of several key aspects:

*   **Language Model Selection**: Choosing an appropriate base or instruction-tuned LLM as the foundation for our agent.
*   **Search/Retrieval Tool**: Determining the nature of the search tool – whether to use a live search engine or a retrieval system built upon a curated document corpus.
*   **Reinforcement Learning Strategy**: Implementing and experimenting with suitable RL algorithms (such as GRPO, Reinforce++, Dr. GRPO, DAPO) to optimize agent behavior.
*   **Comprehensive Evaluation System**: A flexible benchmarking framework that supports any dataset, pipeline, or metric with automatic ground-truth verification and integrated metrics computation.

## 🚀 Evaluation & Benchmarking

Slug-Search includes a comprehensive evaluation system that enables rigorous assessment of both generation quality and retrieval accuracy:

### ✅ **System Flexibility**
- **Dataset Support**: Works with datasets that have compatible structure (see [`slug_search/README.md`](slug_search/README.md) for details)
- **Any Pipeline**: Supports custom RAG pipelines with automatic parameter handling
- **Any Metric**: Dynamic discovery of metrics - just add functions and they're available

### 📊 **Integrated Metrics**
- **Generation Metrics**: Multi-ground-truth answer verification
- **Retrieval Metrics**: Hit rate, precision, and count of ground-truth chunks retrieved
- **Ground-Truth Analysis**: Automatic verification of whether correct chunks were retrieved

**📚 Documentation:**
- **System Overview & Quick Start**: [`slug_search/README.md`](slug_search/README.md) - Complete system documentation with setup guide
- **Evaluation Guide**: [`slug_search/benchmarks/BENCHMARKING_GUIDE.md`](slug_search/benchmarks/BENCHMARKING_GUIDE.md) - Comprehensive evaluation usage

Slug-Search aims to contribute to the understanding of how LLMs can be trained to be more effective and autonomous agents in information-seeking tasks. It serves as a platform for learning and experimentation in advanced RAG techniques, tool use, and applied reinforcement learning.

---
*For details on the original ART (Agent Reinforcement Trainer) framework that this project extends, please see [ART_README.md](./ART_README.md).*