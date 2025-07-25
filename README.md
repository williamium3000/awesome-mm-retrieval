# Awesome-Multi-Modal-Retrieval
> A curated list of Multi-Modal Retrieval methods, datasets, and evaluation benchmarks.

## Table of Contents
- [Awesome-Multi-Modal-Retrieval](#awesome-multi-modal-retrieval)
  - [Table of Contents](#table-of-contents)
  - [Multi-Modal Retrieval Methods](#-multi-modal-retrieval-methods)
    - [Unified Multi-Modal Embedding Model](#unified-multi-modal-embedding-model)
    - [Map to One Unified Modality Before Embedding](#map-to-one-unified-modality-before-embedding)
    - [Bind Multiple Embedding Spaces](#bind-multiple-embedding-spaces)
    - [Other Methods](#other-methods)
    - [Paper List](#paper-list)
  - [Datasets](#-datasets)
  - [Evaluation Benchmarks](#-evaluation-benchmarks)

## Multi-Modal Retrieval Methods

### Unified Multi-Modal Embedding Model
| Method | Description | Paper |
|--------|-------------|-------|
| VLM2Vec-V2 | Unified framework for learning embeddings across text, image, video, and visual document inputs | [VLM2Vec-V2](https://arxiv.org/pdf/2507.04590) |

### Map to One Unified Modality Before Embedding
| Method | Description | Paper |
|--------|-------------|-------|
| UniSE (Universal Screenshot Embeddings) | Convert all multimodal information into unified visual format (screenshots) for retrieval | [VisIR](https://arxiv.org/pdf/2502.11431) |

### Bind Multiple Embedding Spaces
| Method | Description | Paper |
|--------|-------------|-------|
| Cross-modal binding | Learn mappings between different modality-specific embedding spaces | - |

### Other Methods
| Method | Description | Paper |
|--------|-------------|-------|
| MMSEARCH-ENGINE | Multimodal AI search engine pipeline enabling LMMs with search capabilities through requery, rerank, and summarization | [MMSEARCH](https://arxiv.org/pdf/2409.12959) |

### Paper List

<details>
  <summary>VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents</summary>

  [Paper](https://arxiv.org/pdf/2507.04590) | [Project](https://tiger-ai-lab.github.io/VLM2Vec/)
  
  1. **Unified Framework**: Proposes VLM2Vec-V2, a general-purpose embedding model supporting text, image, video, and visual document inputs in a unified framework.
  2. **MMEB-V2 Benchmark**: Introduces comprehensive benchmark extending MMEB with five new task types: visual document retrieval, video retrieval, temporal grounding, video classification, and video QA.
  3. **Cross-Modal Performance**: Achieves strong performance on newly introduced video and document retrieval tasks while improving over prior baselines on original image benchmarks.
  4. **Real-world Applications**: Enables AI agents, multi-modal search and recommendation, and retrieval-augmented generation (RAG) across diverse visual forms.
  
</details>

<details>
  <summary>Visualized Information Retrieval: Unifying Search With Screenshots</summary>

  [Paper](https://arxiv.org/pdf/2502.11431)
  
  1. **VisIR Paradigm**: Formally defines Visualized Information Retrieval where multimodal information (texts, images, tables, charts) is unified into screenshots for retrieval.
  2. **VIRA Dataset**: Creates large-scale dataset with screenshots from diverse sources in captioned and question-answer formats.
  3. **UniSE Model**: Develops Universal Screenshot Embeddings enabling screenshots to query or be queried across arbitrary data modalities.
  4. **MVRB Benchmark**: Constructs Massive Visualized IR Benchmark covering variety of task forms and application scenarios.
  5. **Unified Visual Format**: Treats screenshots as unified entities representing mixture of multimodal data for flexible real-world retrieval.
  
</details>

<details>
  <summary>MMSEARCH: Unveiling the Potential of Large Models as Multi-Modal Search Engines</summary>

  [Paper](https://arxiv.org/pdf/2409.12959) | [Project](https://mmsearch.github.io)
  
  1. **First Multimodal AI Search Engine**: Designs MMSEARCH-ENGINE pipeline to empower any LMMs with multimodal search capabilities, going beyond text-only AI search engines.
  2. **Comprehensive Evaluation**: Introduces MMSEARCH benchmark with 300 manually collected instances spanning 14 subfields for assessing multimodal search performance.
  3. **Three-Stage Pipeline**: Evaluates LMMs through individual tasks (requery, rerank, summarization) and challenging end-to-end complete searching process.
  4. **Performance Analysis**: GPT-4o with MMSEARCH-ENGINE achieves best results, surpassing commercial Perplexity Pro in end-to-end tasks.
  5. **Human-Internet Interaction**: Addresses multimodal user queries and text-image interleaved nature of website information for new paradigm in search.
  
</details>

## Datasets

### Training Datasets
| Dataset | Source | Description | Modalities | Size |
|---------|---------|-------------|------------|------|
| VIRA | [VisIR](https://arxiv.org/pdf/2502.11431) | Large-scale screenshots from diverse sources in captioned and QA formats | Screenshots (Text+Image+Table+Chart) | Large-scale |
| MMSEARCH | [MMSEARCH](https://arxiv.org/pdf/2409.12959) | Manually collected instances spanning 14 subfields for multimodal search | Text + Image | 300 instances |

### Evaluation Datasets
The following table shows the statistics of MMEB-V2 benchmark, which includes 42 tasks across five meta-task categories. The benchmark covers four modalities: T (Text), I (Image), V (Video), and D (Visual Document).

| Task | Query MOD | Target MOD | Domain | #Query | #Candidates |
|------|-----------|------------|---------|---------|-------------|
| **Video Retrieval (5 Tasks)** |
| DiDeMo | T | V | Open | 1,004 | 1,004 |
| MSR-VTT | T | V | Open | 1,000 | 1,000 |
| MSVD | T | V | Open | 670 | 670 |
| VATEX | T | V | Open | 4,468 | 4,468 |
| YouCook2 | T | V | Cooking | 3,179 | 3,179 |
| **Moment Retrieval (3 Tasks)** |
| QVHighlights | T + V | V | Vlog/News | 1,083 | 10 |
| Charades-STA | T + V | V | Activity | 727 | 10 |
| MomentSeeker | T + V | V | Open | 1,800 | 10 |
| **Video Classification (5 Tasks)** |
| Kinetics-700 | V | T | Open | 1,000 | 700 |
| SSv2 | V | T | Human-Object Interaction | 1,000 | 174 |
| HMDB51 | V | T | Open | 1,000 | 51 |
| UCF101 | V | T | Open | 1,000 | 101 |
| BreakfastAI | V | T | Cooking | 453 | 10 |
| **Video QA (5 Tasks)** |
| MVBench | V + T | T | Spatial/Temporal | 4,000 | 3 ~ 5 |
| Video-MME | V + T | T | Real-world | 900 | 4 |
| NExT-QA | V + T | T | Daily activity | 8,564 | 5 |
| EgoSchema | V + T | T | Egocentric | 500 | 5 |
| ActivityNetQA | V + T | T | Activity | 1000 | 2 |
| **Visual Document Retrieval (24 Tasks)** |
| ViDoRe (10) | T | D | Documents | 280 - 1,646 | 70 - 999 |
| ViDoRe-V2 (4) | T | D | Documents | 52 - 640 | 452 - 1,538 |
| VisRAG (6) | T | D | Documents | 63 - 816 | 500 - 3,590 |
| ViDoSeek (2) | T | D | Documents | 1,142 | 5,349 |
| MMDocBench-Doc (2) | T | D | Documents | 838 | 6,492 |

Table 1: Expanding on the statistics of MMEB-V2. ([Source](https://arxiv.org/pdf/2507.04590))

## Evaluation Benchmarks

| Benchmark | Tasks | Coverage | Source | Description |
|-----------|-------|----------|---------|-------------|
| MMEB-V2 | 42 tasks across 5 categories | Text, Image, Video, Visual Document | [VLM2Vec-V2](https://arxiv.org/pdf/2507.04590) | Comprehensive benchmark extending MMEB with video retrieval, temporal grounding, video classification, video QA, and visual document retrieval |
| MVRB | Multiple task forms | Screenshots across modalities | [VisIR](https://arxiv.org/pdf/2502.11431) | Massive Visualized IR Benchmark covering variety of application scenarios for screenshot-based retrieval |
| MMSEARCH | 3 individual + 1 end-to-end | Multimodal search evaluation | [MMSEARCH](https://arxiv.org/pdf/2409.12959) | Evaluation through requery, rerank, summarization tasks and complete searching process across 14 subfields |

<!-- Template for adding new papers -->
<!-- 
<details>
  <summary>Paper Title</summary>

  [Paper](link) | [Github](link) | [Project](link)
  
  Brief description of the method and contributions.
  
</details>
-->
