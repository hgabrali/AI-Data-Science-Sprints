<div align="center">

<!-- Logo / Banner -->
<img src="https://raw.githubusercontent.com/hgabrali/AI-Data-Science-Sprints/main/assets/banner.png" alt="AI Data Science Sprints Banner" width="200"/>

# 🚀 AI-Data-Science-Sprints

### *A Comprehensive Guide to Modern AI-Driven Data Science Workflows*

> **Three-sprint curriculum integrating LLMs, AutoML, and Multimodal AI  
> into every phase of the data science lifecycle.**

---

<!-- Badges -->
![GitHub Stars](https://img.shields.io/github/stars/hgabrali/AI-Data-Science-Sprints?style=for-the-badge&color=yellow)
![GitHub Forks](https://img.shields.io/github/forks/hgabrali/AI-Data-Science-Sprints?style=for-the-badge&color=blue)
![GitHub Issues](https://img.shields.io/github/issues/hgabrali/AI-Data-Science-Sprints?style=for-the-badge&color=red)
![License](https://img.shields.io/github/license/hgabrali/AI-Data-Science-Sprints?style=for-the-badge&color=green)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)

---

<!-- Tech Stack Badges -->
![LLMs](https://img.shields.io/badge/LLMs-GPT--4o%20%7C%20Claude%20%7C%20Llama3-blueviolet?style=flat-square&logo=openai)
![AutoML](https://img.shields.io/badge/AutoML-PyCaret%20%7C%20AutoGluon%20%7C%20H2O-orange?style=flat-square)
![Multimodal](https://img.shields.io/badge/Multimodal-Vision%20%7C%20DALL--E%20%7C%20SAM-teal?style=flat-square)
![Notebooks](https://img.shields.io/badge/Jupyter-Notebooks-orange?style=flat-square&logo=jupyter)

---

<!-- Sprint Overview Pills -->
| 🏁 Sprint 1 | ⚙️ Sprint 2 | 🧠 Sprint 3 |
|:-----------:|:-----------:|:-----------:|
| **Foundations & Tools** | **Data Prep & AutoML** | **Advanced AI Apps** |
| AI Web Search · Image AI · Pandas AI | Auto EDA · Feature Eng. · AutoML | LLM NLP · Code Gen · Multimodal |

---

📖 [Documentation](#overview) · 🐛 [Report Bug](https://github.com/hgabrali/AI-Data-Science-Sprints/issues) · 💡 [Request Feature](https://github.com/hgabrali/AI-Data-Science-Sprints/issues) · ⭐ [Star this Repo](https://github.com/hgabrali/AI-Data-Science-Sprints/stargazers)

</div>

---





# AI-Data-Science-Sprints

> **A Comprehensive Guide to Modern AI-Driven Data Science Workflows**

This repository provides a structured, three-sprint curriculum that integrates cutting-edge AI tools into every phase of the data science lifecycle. Each sprint combines foundational concepts with hands-on examples, leveraging recent advancements in large language models (LLMs), automated machine learning (AutoML), and multimodal AI. The content is designed to be used as both a learning resource and a template for building AI-augmented data science projects.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Sprint 1 – Foundations and Tools of AI in Data Science](#sprint-1--foundations-and-tools-of-ai-in-data-science)
   - [1.1 Introducing AI in Data Science](#11-introducing-ai-in-data-science)
   - [1.2 AI-Based Web Search as the First Step](#12-ai-based-web-search-as-the-first-step)
   - [1.3 AI-Based Assistance for Image Processing](#13-ai-based-assistance-for-image-processing)
   - [1.4 AI-Based Pandas](#14-ai-based-pandas)
5. [Sprint 2 – AI for Data Preparation and Model Building](#sprint-2--ai-for-data-preparation-and-model-building)
   - [2.1 Automated Exploratory Data Analysis (EDA)](#21-automated-exploratory-data-analysis-eda)
   - [2.2 Automated Feature Extraction](#22-automated-feature-extraction)
   - [2.3 Automated Model Training](#23-automated-model-training)
   - [2.4 The Project](#24-the-project)
6. [Sprint 3 – Advanced AI Applications](#sprint-3--advanced-ai-applications)
   - [3.1 Out-of-the-Box NLP with LLMs](#31-out-of-the-box-nlp-with-llms)
   - [3.2 Semi-Automated Code Writing](#32-semi-automated-code-writing)
   - [3.3 Not Only Texts but Also Visuals](#33-not-only-texts-but-also-visuals)
   - [3.4 The Project](#34-the-project)
7. [References & Further Reading](#references--further-reading)

---

## Overview

Modern data science is no longer a purely manual process. AI tools now assist or automate everything from data discovery and cleaning to feature engineering, model selection, and even report generation. This repository explores three sprints that progressively introduce AI-powered techniques:

- **Sprint 1** builds a foundation: using AI for initial data exploration, web search, image processing, and data manipulation.
- **Sprint 2** focuses on automation: automated EDA, feature extraction, and model training with AutoML.
- **Sprint 3** dives into advanced applications: leveraging large language models for NLP, code generation, and multimodal visualisation.

Each section includes code examples (Python), references to state-of-the-art libraries, and discussions of recent developments (e.g., GPT-4o, multimodal LLMs, agentic workflows).

---

## Repository Structure

```text
AI-Data-Science-Sprints/
├── README.md                 # This file
├── requirements.txt          # Common dependencies
├── sprint1/
│   ├── 01_intro_ai_ds.ipynb
│   ├── 02_ai_web_search.ipynb
│   ├── 03_ai_image_processing.ipynb
│   ├── 04_ai_pandas.ipynb
│   └── data/                 # Sample datasets
├── sprint2/
│   ├── 01_automated_eda.ipynb
│   ├── 02_automated_features.ipynb
│   ├── 03_automated_training.ipynb
│   └── project/              # Sprint 2 project scaffold
├── sprint3/
│   ├── 01_llm_nlp.ipynb
│   ├── 02_semi_auto_code.ipynb
│   ├── 03_visuals.ipynb
│   └── project/              # Sprint 3 project scaffold
└── utils/                    # Helper functions and API wrappers
```

---

## Prerequisites

- Python 3.9+
- Basic knowledge of Python and data science libraries (pandas, numpy, matplotlib)
- API keys for OpenAI, Anthropic, or other LLM providers (if you wish to run the live examples)
- A working Jupyter environment (JupyterLab, VS Code, or Google Colab)

Install the core dependencies with:

```bash
pip install -r requirements.txt
```

---

## Sprint 1 – Foundations and Tools of AI in Data Science

This sprint introduces AI-assisted tools that accelerate the early stages of a data science project.

### 1.1 Introducing AI in Data Science

**Objective:** Understand how AI is transforming traditional data science workflows.

#### Key Concepts

- **AI-augmented data science:** LLMs can generate code, explain results, and suggest next steps.
- **Agentic workflows:** AI agents that combine tools (e.g., search, code execution) to perform multi-step tasks.
- **Ethical considerations:** bias, transparency, and data privacy when using AI tools.

#### Recent Developments

- GPT-4o and Claude 3.5 support native tool calling, enabling agents to use pandas, matplotlib, and external APIs.
- OpenAI's Assistants API and LangChain allow building AI assistants tailored for data science.

#### Example: Using an LLM to explain a dataset

```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are a data science expert."},
        {"role": "user", "content": "Describe what kind of analysis I should perform on a customer churn dataset."}
    ]
)
print(response.choices[0].message.content)
```

---

### 1.2 AI-Based Web Search as the First Step

**Objective:** Use AI to gather relevant data, research, and documentation automatically.

#### Tools & Techniques

- **LLM + Search API:** Combine a language model with search engines (Google, Bing, SerpAPI) to fetch and summarise information.
- **Web scraping with AI:** Use models like GPT-4-vision to parse complex web pages or Firecrawl for structured extraction.
- **Automated literature review:** Tools like Elicit or Scite leverage LLMs to summarise research papers.

#### Recent Developments

- Perplexity AI-style search agents can be built using LangChain's `create_react_agent`.
- Google Gemini integrates directly with Google Search for real-time data retrieval.

#### Example: AI-powered search for dataset discovery

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
tools = [Tool(
    name="Search",
    func=search.run,
    description="Useful for finding datasets and documentation"
)]

# Full agent setup available in sprint1/02_ai_web_search.ipynb
```

---

### 1.3 AI-Based Assistance for Image Processing

**Objective:** Leverage multimodal AI to analyse, process, and generate insights from images.

#### Key Libraries

| Library | Purpose |
|---|---|
| OpenCV + CLIP | Image classification and natural-language image search |
| Segment Anything Model (SAM) | Zero-shot segmentation (Meta AI) |
| Stable Diffusion / DALL-E | Text-to-image generation and editing |
| GPT-4V (Vision) | Image description, OCR, visual Q&A |

#### Recent Developments

- **Florence-2** (Microsoft) unifies object detection, captioning, and segmentation in one model.
- **IDEFICS** and **LLaVA** are open-source multimodal models that run fully locally.

#### Example: Image captioning with a multimodal LLM

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]}
    ]
)
print(response.choices[0].message.content)
```

---

### 1.4 AI-Based Pandas

**Objective:** Use AI to write, explain, and optimise pandas code automatically.

#### Tools

- **PandasAI:** Enables natural language queries directly on DataFrames.
- **Jupyter AI:** Adds conversational AI cells in Jupyter notebooks.
- **DataWizard (Hugging Face):** Auto-generates pandas transformation code from text.

#### Recent Developments

- **Google's DataGemma:** A specialised model for data analysis and table understanding.
- **SQL-to-pandas using LLMs:** Translate SQL queries into equivalent pandas operations at runtime.

#### Example: Natural language to pandas with PandasAI

```python
from pandasai import SmartDataframe
import pandas as pd

df = pd.read_csv("sales.csv")
sdf = SmartDataframe(df)
result = sdf.chat("What is the total sales per region?")
print(result)
```

---

## Sprint 2 – AI for Data Preparation and Model Building

This sprint focuses on automation – letting AI handle the repetitive parts of data preparation and modelling.

### 2.1 Automated Exploratory Data Analysis (EDA)

**Objective:** Generate comprehensive EDA reports with a single line of code.

#### Tools

| Tool | Highlights |
|---|---|
| ydata-profiling (formerly pandas-profiling) | Interactive HTML reports with statistics and visualisations |
| Sweetviz | Side-by-side train vs. test dataset comparison |
| Dataprep.eda | Fast, scalable, visual EDA |
| Lux | Intent-driven visualisation recommendations |

#### Recent Developments

- **LLM-powered EDA:** Tools like EDA-GPT automatically generate insights and narratives from profiling output.
- **Auto-Insights:** Microsoft's Data-Insights library uses LLMs to highlight anomalies and correlations.

#### Example: Generate a profiling report

```python
from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("dataset.csv")
profile = ProfileReport(df, title="Profiling Report", explorative=True)
profile.to_file("report.html")
```

---

### 2.2 Automated Feature Extraction

**Objective:** Automatically create features from raw data, especially for time series, relational, and text data.

#### Tools

- **Featuretools:** Deep Feature Synthesis (DFS) for relational datasets.
- **tsfresh:** Automatic time-series feature extraction with statistical tests.
- **Hugging Face Transformers:** Embed text using pre-trained models (BERT, RoBERTa, etc.).
- **LLM-based feature generation:** Prompt GPT to suggest or code domain-specific features.

#### Recent Developments

- **AutoFeat:** Sklearn-compatible automated feature engineering.
- **LLM as a feature engineer:** Agentic pipelines prompt models to suggest features and then execute the generated code.

#### Example: Deep Feature Synthesis with Featuretools

```python
import featuretools as ft

es = ft.EntitySet(id="customer_data")
es = es.add_dataframe(dataframe=df, dataframe_name="customers", index="id")

feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="customers",
    max_depth=2
)
```

---

### 2.3 Automated Model Training

**Objective:** Use AutoML to train, tune, and select models with minimal manual intervention.

#### Tools

| Library | Strengths |
|---|---|
| PyCaret | Low-code, end-to-end ML pipeline with built-in explainability |
| H2O AutoML | Distributed, leaderboard-based AutoML with stacking |
| AutoGluon (Amazon) | State-of-the-art ensembling; supports tabular, text, and image |
| FLAML | Fast, lightweight AutoML from Microsoft Research |

#### Recent Developments

- **AutoML 2.0:** Integration with LLMs to explain model decisions and suggest hyperparameters.
- **Multi-modal AutoML:** AutoGluon now jointly trains on text, image, and tabular data.

#### Example: PyCaret for binary classification

```python
from pycaret.classification import setup, compare_models, pull, save_model

clf = setup(data=df, target="churn", session_id=42, normalize=True)
best_model = compare_models(n_select=3)

# Display leaderboard
leaderboard = pull()
print(leaderboard.head())

# Save best model
save_model(best_model, "best_churn_model")
```

---

### 2.4 The Project

**Objective:** Apply automated EDA, feature extraction, and AutoML to a real-world dataset end-to-end.

#### Project Outline

1. **Choose a dataset** – e.g., Kaggle's House Prices, Telco Customer Churn, or the UCI Adult dataset.
2. **Automated EDA** – Generate a ydata-profiling report and use an LLM to summarise key findings.
3. **Feature extraction** – Apply Featuretools DFS or tsfresh (if time-series data is present).
4. **AutoML comparison** – Benchmark PyCaret, AutoGluon, and H2O AutoML on the same train/test split.
5. **Model interpretation** – Explain the best model using SHAP values and LIME.
6. **Final report** – Write a short narrative summarising findings, metrics, and insights.

#### Deliverables

- Jupyter notebook with reproducible code and commentary.
- HTML EDA report (committed to the `project/` folder).
- Model leaderboard screenshots and performance metrics (ROC-AUC, F1, RMSE as applicable).
- Reflection document: which AI tools saved the most time and what pitfalls were encountered.

---

## Sprint 3 – Advanced AI Applications

This sprint explores cutting-edge AI that goes beyond traditional ML, including LLMs for NLP, code generation, and multimodal creativity.

### 3.1 Out-of-the-Box NLP with LLMs

**Objective:** Use pre-trained LLMs for complex NLP tasks without fine-tuning.

#### Key Tasks

- Text classification, sentiment analysis, and named entity recognition via prompt engineering.
- **Retrieval-Augmented Generation (RAG):** Combine LLMs with external knowledge bases (company documents, research papers).
- **Text-to-SQL:** Convert natural language questions to SQL queries.

#### Recent Developments

- **Function calling** (OpenAI, Anthropic) allows LLMs to reliably output structured JSON data.
- **Local models:** Llama 3, Mistral, and Qwen can run on consumer hardware using Ollama or vLLM.
- **RAG frameworks:** LangChain, LlamaIndex, and Haystack simplify building production-ready pipelines.

#### Example: Sentiment analysis via prompt engineering

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are a sentiment analyzer. Output only 'positive', 'negative', or 'neutral'."
        },
        {
            "role": "user",
            "content": "The product quality is excellent, but shipping was disappointingly slow."
        }
    ]
)
print(response.choices[0].message.content)
```

#### Example: Simple RAG pipeline with LlamaIndex

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents from a local folder
documents = SimpleDirectoryReader("./docs").load_data()

# Build index and query
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What are the main findings in the Q3 report?")
print(response)
```

---

### 3.2 Semi-Automated Code Writing

**Objective:** Use AI to write, refactor, test, and document code.

#### Tools

- **GitHub Copilot:** AI pair programmer integrated into VS Code and JetBrains IDEs.
- **Cursor:** AI-native code editor with multi-file context and autonomous editing.
- **Code generation models:** CodeLlama, StarCoder2, GPT-4, Claude 3.5 Sonnet.
- **LLM-based code review:** Automatically surface improvements, bugs, and style issues.

#### Recent Developments

- **Agentic code writing:** Agents (e.g., Devin, SWE-agent) write tests, run them, and iterate until all tests pass.
- **Documentation generation:** Tools like Mintlify and Swimm use LLMs to create docstrings, README files, and API references automatically.

#### Example: Generate a utility function with an LLM

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Write clean, type-annotated, well-documented Python code."},
        {
            "role": "user",
            "content": (
                "Write a function that computes the rolling average of a pandas Series "
                "with a configurable window size, handling NaN values gracefully. "
                "Include a docstring and a usage example."
            )
        }
    ]
)
print(response.choices[0].message.content)
```

---

### 3.3 Not Only Texts but Also Visuals

**Objective:** Generate and interpret visual content using multimodal AI.

#### Techniques

- **Text-to-image:** DALL-E 3, Stable Diffusion XL, Midjourney.
- **Text-to-chart:** LLMs generate matplotlib or Plotly code from natural-language chart descriptions.
- **Image editing:** Inpainting, outpainting, and style transfer with diffusion models.
- **AI-driven dashboards:** Prompt LLMs to scaffold full interactive dashboards (Streamlit, Dash).

#### Recent Developments

- **Multimodal generation:** Sora (text-to-video) and Suno (text-to-audio) expand storytelling possibilities in data presentations.
- **LLM-driven dashboard generation:** Generate complete Streamlit apps from a single natural-language prompt.

#### Example: Generate a chart from a natural-language description

```python
from openai import OpenAI

client = OpenAI()

prompt = (
    "Write Python matplotlib code to create a horizontal bar chart "
    "showing the top 5 product categories by total sales. "
    "Include a descriptive title, axis labels, value annotations on each bar, "
    "and use a colour-blind-friendly palette. Return only the executable code block."
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

generated_code = response.choices[0].message.content
print(generated_code)
# exec(generated_code)  # Uncomment only in a trusted, sandboxed environment
```

---

### 3.4 The Project

**Objective:** Build an end-to-end AI-powered application combining NLP, code generation, and visual output.

#### Project Ideas

**Option A – AI Research Assistant**
- Use web search + LLM to fetch academic papers, summarise abstracts, extract key findings, and generate a structured literature review with visual citation maps.

**Option B – Interactive Data Storyteller**
- Accept an arbitrary dataset and a business question; automatically run exploratory analysis, write a narrative summary, and generate publication-ready charts using LLM-produced code.

**Option C – Multimodal Content Creator**
- Input a topic; use an LLM to write a technical blog post, generate supporting images with DALL-E, and produce a polished HTML report with embedded figures.

#### Deliverables

- A working Python script or Jupyter notebook demonstrating the chosen application end-to-end.
- A short screen-recording or slide deck explaining the workflow and the role of AI at each step.
- Written discussion of current limitations, failure modes, and proposed future improvements.

---

## References & Further Reading

| Resource | Link |
|---|---|
| PandasAI Documentation | https://docs.pandas-ai.com |
| PyCaret | https://pycaret.org |
| LangChain | https://python.langchain.com |
| LlamaIndex | https://docs.llamaindex.ai |
| AutoGluon | https://auto.gluon.ai |
| OpenAI Function Calling | https://platform.openai.com/docs/guides/function-calling |
| Segment Anything Model (SAM) | https://github.com/facebookresearch/segment-anything |
| ydata-profiling | https://docs.profiling.ydata.ai |
| Featuretools | https://www.featuretools.com |
| H2O AutoML | https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html |
| Hugging Face Transformers | https://huggingface.co/docs/transformers |
| FLAML | https://microsoft.github.io/FLAML |

---

## License

This repository is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the content with attribution.

---

*Built for the data science community. Contributions, issues, and pull requests are welcome!*
