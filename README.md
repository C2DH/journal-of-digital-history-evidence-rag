> 🔒 **Confidentiality Note:** All sensitive information, including paper titles, paper content, and authors’ names, has been masked.

# 🧾 Evidence-RAG for Peer Review Auditing in the Journal of Digital History

An evidence-grounded **Retrieval-Augmented Generation (RAG)** workspace for [Journal of Digital History (JDH)](https://journalofdigitalhistory.org/en/) papers.  
The system supports semantic chunking, vector indexing, retrieval, and LLM-based inference for paper-level review auditing.

## 📌 Project Overview

This project builds an **Evidence-RAG system for JDH review auditing**.

The main goal is to connect each reviewer comment to specific evidence from the submitted paper. This helps editors see whether a reviewer’s claim is:

- ✅ **Supported** by the paper text
- ⚠️ **Partially supported** by the paper text
- ❌ **Not supported** by the paper text

The system does **not** replace the editor’s judgment. Instead, it provides a transparent evidence layer that helps editors evaluate reviewer comments more systematically, consistently, and explainably.

### Pre-requisites

For this project, we use the **Grid5000** infrastructure to run our models. You can find information about registration [here](https://www.grid5000.fr/w/Grid5000:Get_an_account). Grid5000 can only be used by **academicians or people affiliated to university**.

### Quick Start

In this repository, you have two options:

1. **Option 1:** You can view and use the **Jupyter notebooks** available in the `notebooks` directory. To run notebooks 1 and 3, we recommend having a Grid5000 account. See the pre-requisites section above for more information.

2. **Option 2:** The same code is also available in the `source` directory, which allows you to run the models locally.

## 🧭 Workflow Summary

The complete workflow is organized into four main notebooks:

1. 📥 **Paper collection and text conversion**
2. 📝 **Review file processing**
3. 🧠 **Vector database and evidence preparation**
4. 🔎 **Final Evidence-RAG review audit**

Each notebook performs one important step in the full pipeline.

The main models used or compared in this project are:

- 🤖 **Qwen/Qwen2.5-32B-Instruct**  
  Hugging Face: <https://huggingface.co/Qwen/Qwen2.5-32B-Instruct>

- 🤖 **Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF**  
  Hugging Face: <https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled>

- 🧬 **BAAI/bge-large-en-v1.5** for text embeddings  
  Hugging Face: <https://huggingface.co/BAAI/bge-large-en-v1.5>

## 📥 1. Paper Collection and Text Conversion

**Notebook file:**  
`0_JDH_Read_jupyterNotebook_Papers_Convert_to_Text (6) (1).ipynb`

In this notebook, we collect the JDH article notebooks and convert them into readable text files.

This step extracts article content from the original notebook format, including:

- 📄 Markdown cells
- 💻 Code cells
- 📊 Relevant outputs when needed
- 🏷️ Paper metadata

The output of this step is a cleaned text version of each paper. These cleaned files are later used for chunking, embedding, retrieval, and evidence checking.

## 📝 2. Review File Processing

**Notebook file:**  
`1_JDH_Review_jupyterNotebook (oss_120b_ProcessReviewFiles) (5) (2).ipynb`

In this notebook, we process the reviewer PDF files.

We use **OpenAI GPT-OSS-120B** on **Grid5000** to read the review files and split them into structured reviewer comments.

The output contains separated reviewer comments with information such as:

- 🆔 Paper ID
- 👤 Reviewer name or reviewer label
- 🔁 Review round
- 🔢 Comment ID
- 💬 Individual reviewer comment text

This step is important because each reviewer comment must be checked separately against the paper evidence.

## 🧠 3. Vector Database and Paper Evidence Preparation

**Notebook file:**  
`2_JDH_VectorDatabase_jupyterNotebook (ProcesPapersFiles) (5) (2).ipynb`

In this notebook, we prepare the searchable paper evidence database.

The paper texts are divided into meaningful semantic chunks using:

- 🤖 **Qwen/Qwen2.5-32B-Instruct** for semantic chunking
- 🧬 **BAAI/bge-large-en-v1.5** for text embeddings
- 🗄️ **Qdrant** as the vector database

Each paper has its own Qdrant collection. Each chunk keeps useful metadata, including:

- 🆔 Paper ID
- 📁 Source file
- 📚 Section title
- 🔢 Chunk ID
- 🏷️ Chunk label
- 🧾 Chunk summary
- 📄 Chunk text

This step creates the evidence base that the Evidence-RAG system searches when evaluating reviewer comments.

## 🔎 4. Final Evidence-RAG Review Audit

**Notebook file:**  
`3_JDH_EvidenceRAG_jupyterNotebook _Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF(Final Review Auditor Grid5000) (6) (1).ipynb`

In this notebook, we run the final Evidence-RAG audit.

For each reviewer comment, the system retrieves the most relevant paper chunks from the Qdrant vector database. Then, the model evaluates whether the reviewer comment is supported by the retrieved evidence.

The final output includes:

- 💬 Reviewer comment ID
- 📚 Retrieved evidence chunks
- ✅ Support label
- 🧠 Reasoning summary
- 📈 Retrieval strength
- 🤖 Model support strength
- 🎯 Evidence specificity
- 🧩 Section coverage
- 📊 Final confidence score

This final audit file gives editors a structured view of how each reviewer comment relates to the actual paper content.

## 📊 Confidence Score Formula

The final confidence score combines four components:

```text
Confidence = 0.15 × Retrieval Strength
           + 0.25 × Model Support Strength
           + 0.45 × Evidence Specificity
           + 0.15 × Section Coverage
```

## 🧮 Meaning of Each Confidence Component

### 📈 Retrieval Strength

**Retrieval Strength** measures how strongly the vector database retrieved relevant chunks for the reviewer comment.

It is based on the similarity scores of the retrieved evidence chunks.

### 🤖 Model Support Strength

**Model Support Strength** measures how confidently the model judges that the reviewer comment is supported by the retrieved evidence.

This score reflects the model’s assessment of the relationship between the reviewer comment and the retrieved chunks.

### 🎯 Evidence Specificity

**Evidence Specificity** measures how directly the retrieved chunks address the reviewer comment.

This is important because a retrieved chunk may be generally related to the paper but not specific enough to support the exact reviewer claim.

### 🧩 Section Coverage

**Section Coverage** measures how broadly the retrieved evidence covers different sections of the paper.

For example, if a paper has five sections and the retrieved evidence comes from two unique sections, then:

```text
Section Coverage = 2 / 5 = 0.40
```

This score helps show whether the evidence is concentrated in one part of the paper or spread across multiple relevant sections.

## 🧑‍⚖️ Role of the Editor

The system does **not** make the final editorial decision.

Instead, it supports the editor by showing:

- 📌 Which paper passages were retrieved
- ✅ Whether the model thinks the reviewer comment is supported
- 🧠 Why the model made that judgment
- 📊 How the confidence score was calculated
- 🗂️ Which sections of the paper were used as evidence

The editor remains the final human decision-maker.

## 🎯 Final Purpose

The purpose of this workflow is to make the review-auditing process more:

- 🔍 Transparent
- 📚 Evidence-based
- 🧩 Systematic
- 🧠 Explainable
- 🧑‍⚖️ Editor-friendly

More specifically, the system aims to:

- 🚀 Generate rapid, structured feedback that helps authors improve their papers when reviewer capacity is limited
- 🤝 Support multi-agent and multi-role review workflows that can be compared with human judgments
- 📖 Offer guided reading by showing the exact passages and artifacts used as evidence
- 🧪 Help check consistency between reported methods, supplied code, and reviewer comments
- 🗂️ Provide editors with a structured evidence layer for reviewer-comment evaluation

By combining reviewer comments, paper evidence, vector retrieval, and model-based evaluation, the Evidence-RAG system helps editors better understand how strongly each reviewer comment is grounded in the submitted paper.

## ✅ Summary

This project creates a complete Evidence-RAG pipeline for JDH review auditing:

```text
JDH papers
   ↓
Text extraction
   ↓
Semantic chunking
   ↓
Vector database construction
   ↓
Reviewer comment extraction
   ↓
Evidence retrieval
   ↓
Model-based support judgment
   ↓
Editor-facing audit output
```

The final result is an editor-support system that links reviewer claims to paper evidence and explains how strongly each claim is supported.
