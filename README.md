# Awesome-Multi-Modal-Embedding
> A curated list of Multi-Modal embedding and retrieval methods, datasets, and evaluation benchmarks.

## Table of Contents
- [Awesome-Multi-Modal-Embedding](#awesome-multi-modal-embedding)
  - [Table of Contents](#table-of-contents)
  - [Multi-Modal Retrieval](#multi-modal-retrieval)
    - [Unified Multi-Modal Embedding with MLLMs](#unified-multi-modal-embedding-with-mllms)
    - [Map to One Unified Modality Before Embedding](#map-to-one-unified-modality-before-embedding)
    - [Multi-Modal Embedding](#multi-modal-embedding)
    - [Composed Image Retrieval](#composed-image-retrieval)
    - [Agentic and Search](#agentic-and-search)
  - [Multimodal-RAG-Workflow](#multimodal-rag-workflow)
  - [Datasets](#datasets)

## Multi-Modal Retrieval

### Unified Multi-Modal Embedding with MLLMs

<details>
  <summary>E5-V: Universal Embeddings with Multimodal Large Language Model</summary>

  [Paper](https://arxiv.org/abs/2407.12580) | [Github](https://github.com/kongds/E5-V)
  
  TLDR: E5-V proposes a single modality training approach where the model is trained exclusively on text pairs, converting images to text using multimodal LLMs like LLaVA-Next. This method demonstrates significant improvements over traditional multimodal training on image-text pairs while reducing training costs by approximately 95%. The approach is based on SimCSE and Alpaca-LoRA, using Llama3-LLaVA-Next-8B, and achieves strong performance on image-image retrieval tasks by rendering textual captions as images.
  
</details>


<details>
  <summary>VLM2Vec-V2: Unified Framework for Vision-Language Models</summary>

  [Paper](https://arxiv.org/pdf/2507.04590)
  
  TLDR: VLM2Vec-V2 is a unified framework for learning embeddings across text, image, video, and visual document inputs. The model provides a comprehensive approach to multi-modal representation learning that handles diverse input modalities within a single framework.
  
</details>


<details>
  <summary>MCSE: Multimodal Contrastive Learning of Sentence Embeddings</summary>

  [Paper](https://aclanthology.org/2022.naacl-main.436/)
  
  TLDR: MCSE performs multimodal contrastive learning of sentence embeddings with textual and visual alignment. The method leverages contrastive learning to align sentence representations across text and visual modalities, enabling better cross-modal understanding and retrieval.
  
</details>


### Map to One Unified Modality Before Embedding
| Method | Description | Paper |
|--------|-------------|-------|
| UniSE (Universal Screenshot Embeddings) | Convert all multimodal information into unified visual format (screenshots) for retrieval | [VisIR](https://arxiv.org/pdf/2502.11431) |
| Multi-RAG | Convert all modalities to text for adaptive video understanding | [Multi-RAG (JHU 2025)](https://arxiv.org/abs/2501.00000) |
| Graph-based Unified | Convert images to scene graphs and sentences to dependency trees | [CVPR 2019](https://arxiv.org/abs/1904.05521) |

### Multi-Modal Embedding
| Method | Description | Paper |
|--------|-------------|-------|
| OTKGE | Multi-modal knowledge graph embeddings via optimal transport | [OTKGE (NeurIPS 2022)](https://arxiv.org/abs/2206.04611) |


<details>
  <summary>ImageBind: One Embedding Space To Bind Them All</summary>

  [Paper](https://arxiv.org/abs/2305.05665) | [Github](https://github.com/facebookresearch/ImageBind)
  
  TLDR: ImageBind unifies six modalities (image, text, audio, video, thermal, depth, IMU) into a single joint embedding space using only image-paired data and contrastive learning. The method uses modality-specific encoders (ViT for images/videos, spectrograms for audio, 1D convolution for IMU) with InfoNCE loss to align all modalities to images. Once aligned to image embeddings, cross-modal tasks like text-audio retrieval work emergently without requiring cross-modal pairs, demonstrating strong zero-shot classification and modality-compositionality capabilities.
  
</details>


<details>
  <summary>MERL: Multimodal Event Representation Learning in Heterogeneous Embedding Spaces</summary>

  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16270)
  
  TLDR: MERL learns multimodal event representations in heterogeneous embedding spaces where text triples are encoded as Gaussian density embeddings and images are projected into point embedding spaces. The method uses a statistical score function inspired by likelihood-ratio hypothesis testing to align the two spaces, ensuring image embeddings behave as samples drawn from their corresponding text triple's Gaussian distribution. Training combines intra-modal losses for text and images with cross-modal loss to enforce alignment between Gaussian text embeddings and image point embeddings.
  
</details>


<details>
  <summary>UniversalRAG: Retrieval-Augmented Generation over Diverse Modalities and Granularities</summary>

  [Paper](https://arxiv.org/pdf/2504.20734)
  
  TLDR: UniversalRAG avoids using a shared embedding space due to modality gap bias and instead introduces a Router module (using GPT-4o or lightweight models like DistilBERT) to decide which modality-specific corpus to search and at what granularity. The system performs retrieval in modality-specific spaces for different content types (paragraph, document, clip, video) and passes results to the LVLM generator. This approach addresses the challenges of multimodal search across diverse content types while maintaining retrieval quality within each modality.
  
</details>

### Composed Image Retrieval


### Agentic and Search
| Method | Description | Paper |
|--------|-------------|-------|
| MMSEARCH-ENGINE | Multimodal AI search engine pipeline enabling LMMs with search capabilities through requery, rerank, and summarization | [MMSEARCH](https://arxiv.org/pdf/2409.12959) |
| NExT-GPT | Any-to-any multimodal LLM with modality adaptors and diffusion decoders | [NExT-GPT (ICML 2024 Oral)](https://arxiv.org/abs/2309.05519) |
| Vision Search Assistant | Empower VLMs as multimodal search agents using click-based simulations | [Vision Search Assistant](https://arxiv.org/abs/2404.13443) |
| Contrastive Alignment (CAL) | Prioritizing visual correlation by contrastive alignment for re-weighting | [CAL (NeurIPS 2024)](https://arxiv.org/abs/2406.12230) |
| Deep Cross-Modal Hashing | Use deep neural networks to jointly learn representations and hash functions | [Review](https://arxiv.org/abs/2104.13923) |
| Semantic Preserving Hashing | Incorporate label or semantic information to enhance retrieval accuracy | [Survey](https://arxiv.org/abs/2104.13923) |

<details>
  <summary>NExT-GPT: Any-to-Any Multimodal LLM</summary>

  [Paper](https://arxiv.org/abs/2309.05519) | [Project](https://next-gpt.github.io/)
  
  NExT-GPT connects an LLM with multimodal adaptors and different diffusion decoders to enable any-to-any multimodal generation, leveraging existing well-trained high-performing encoders and decoders. The system uses modality-switching instruction tuning on the MosIT dataset containing 5K high-quality dialogues with 3-7 turns each, where interactions involve multiple modalities at either input or output side. The work demonstrates how to effectively combine retrieval systems and AIGC tools like Stable-XL and Midjourney for comprehensive multimodal conversation capabilities.
  
</details>



<details>
  <summary>Vision Search Assistant: Empower Vision-Language Models as Multimodal Search Agents</summary>

  [Paper](https://arxiv.org/abs/2404.13443)
  
  Vision Search Assistant turns VLMs into search agents that can retrieve and ground information in multimodal web pages or documents using click-based simulations and HTML document trees. The system uses vision-language models like BLIP-2 or LLaVA to process webpage screenshots or rendered documents, with a navigation module trained to simulate retrieval actions like clicking or scrolling to locate relevant visual and textual information, implementing document-grounded VQA that mimics human browsing behavior.
  
</details>


## Multimodal RAG Workflow


<details>
  <summary>VisRAG: Vision-Based Retrieval-Augmented Generation on Multi-Modality Documents</summary>

  [Paper](https://arxiv.org/abs/2410.10594)
  
  TLDR: VisRAG embeds documents as images to avoid information loss during parsing, using position-weighted mean pooling on the last hidden layer of vision-language models for retrieval. The system trains a retriever on 280K VQA data using InfoNCE loss and employs frozen VLMs like MiniCPM-V and GPT-4o as generators. For multi-document retrieval, the method handles multiple images through horizontal concatenation for single-image VLMs or direct multi-image input for capable models, achieving 25-39% improvement over pure text RAG in end-to-end evaluation.
  
</details>

<details>
  <summary>RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs</summary>

  [Paper](https://arxiv.org/abs/2407.02485)
  
  TLDR: RankRAG demonstrates that Llama3-RankRAG-8B and Llama3-RankRAG-70B instruction-tuned LLMs work surprisingly well by adding a small fraction of ranking data into the training blend. The approach unifies context ranking with retrieval-augmented generation, showing how ranking capabilities can be effectively integrated into large language models for improved retrieval performance.
  
</details>

<details>
  <summary>Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal LLMs</summary>

  [Paper](https://arxiv.org/abs/2404.15406)
  
  TLDR: Wiki-LLaVA augments LLaVA with a hierarchical external retrieval mechanism over multimodal documents using CLIP encoders for both images and text titles. The system performs two-stage retrieval: first retrieving relevant documents based on similarity to image+question, then retrieving passages from top-k documents using Contriever dense retriever. This hierarchical approach enables effective multimodal document understanding and question answering through structured retrieval processes.
  
</details>

<details>
  <summary>RAR: Retrieving And Ranking Augmented MLLM for Visual Recognition</summary>

  [Paper](https://arxiv.org/abs/2404.13443)
  
  TLDR: RAR leverages external knowledge bases to enhance MLLMs by addressing language ambiguity, synonym handling, and limited context window problems. The system first constructs a multimodal retriever that creates and stores multimodal embeddings of visual images and text descriptions, then during inference retrieves the top-k most similar categories to the input image and uses MLLM ranking of these candidates for final prediction. This multimodal large model fusion with RAG approach activates the image understanding potential of VLMs through external knowledge augmentation.
  
</details>

## Datasets

| Dataset | Source | Description | Modalities | Size |
|---------|---------|-------------|------------|------|
| VIRA | [VisIR](https://arxiv.org/pdf/2502.11431) | Large-scale screenshots from diverse sources in captioned and QA formats | Screenshots (Text+Image+Table+Chart) | Large-scale |
| MMSEARCH | [MMSEARCH](https://arxiv.org/pdf/2409.12959) | Manually collected instances spanning 14 subfields for multimodal search | Text + Image | 300 instances |
| MosIT | [NExT-GPT](https://arxiv.org/abs/2309.05519) | Modality-switching instruction tuning with 3-7 turn conversations | Text + Image + Audio + Video | 5K dialogues |
| VisRAG Collection | [VisRAG](https://arxiv.org/abs/2410.10594) | VQA data with document images avoiding parsing information loss | Text + Visual Documents | 279K (239K VQA + 40K synthetic) |

### Single-Modal Retrieval Datasets

| Dataset | Task Type | Description | Size |
|---------|-----------|-------------|------|
| SQuAD | Text RAG | Single-hop RAG with paragraphs as retrieval units | - |
| Natural Questions (NQ) | Text RAG | Single-hop RAG with paragraphs as retrieval units | - |
| HotpotQA | Text RAG | Multi-hop RAG with documents as retrieval units | - |
| WebQA | Image RAG | Queries requiring grounding in external images | - |
| VideoRAG-Wiki | Video RAG | Queries requiring comprehension of long-form videos | - |
| VideoRAG-Synth | Video RAG | Queries requiring comprehension of complete videos | - |
| LVBench | Video RAG | Queries targeting short or localized video segments | - |

### Multi-Modal Paired Datasets

| Dataset | Modality Pairs | Source | Usage |
|---------|----------------|---------|-------|
| CMU-MOSEI | Video + Audio + Text | YouTube speakers | Human emotion recognition and sentiment analysis (23,500 sentences, 1,000 speakers) |
| Audioset | Video + Audio | - | Training pairs |
| SUN RGB-D | Image + Depth | - | Training pairs (replicated 50×) |
| LLVIP | Image + Thermal | - | Training pairs (replicated 50×) |
| Ego4D | Video + IMU | - | Training pairs |
| Flickr30K (I2I) | Image + Image | Text captions rendered as images | Image-image retrieval |
| COCO (I2I) | Image + Image | Text captions rendered as images | Image-image retrieval |
| DocVQA | Text + Visual Documents | - | Document visual question answering |



### Evaluation Benchmarks

| Benchmark | Tasks | Coverage | Source | Description |
|-----------|-------|----------|---------|-------------|
| MMEB-V2 | 42 tasks across 5 categories | Text, Image, Video, Visual Document | [VLM2Vec-V2](https://arxiv.org/pdf/2507.04590) | Comprehensive benchmark extending MMEB with video retrieval, temporal grounding, video classification, video QA, and visual document retrieval |
| MVRB | Multiple task forms | Screenshots across modalities | [VisIR](https://arxiv.org/pdf/2502.11431) | Massive Visualized IR Benchmark covering variety of application scenarios for screenshot-based retrieval |
| MMSEARCH | 3 individual + 1 end-to-end | Multimodal search evaluation | [MMSEARCH](https://arxiv.org/pdf/2409.12959) | Evaluation through requery, rerank, summarization tasks and complete searching process across 14 subfields |

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

*Table 1: An expansion on the statistics of MMEB-V2.* ([Source](https://arxiv.org/pdf/2507.04590))

<!-- Template for adding new papers -->
<!-- 
<details>
  <summary>Paper Title</summary>

  [Paper](link) | [Github](link) | [Project](link)
  
  Brief description of the method and contributions.
  
</details>
-->
