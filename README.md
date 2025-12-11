# LLM Content Moderation - Artifacts

**CONTENT WARNING:** Both the corpus and result files contain disturbing content related to problematic queries. Reader discretion is strongly advised.

**AUTHOR POSITIONALITY:** All authors and other persons involved as part of the project do not endorse or specify likeness to any of the statements found in this corpus. They do not represent any of the views or opinions held by any persons involved with the project.

**CONTENT REDACTION:** Certain statements are redacted due to their risk to people contacted during the project and sensitive nature, however if future researchers want access to the unredacted dataset they should contact the authors.

## Overview

This paper evaluates the content moderation applied by LLMs to problematic queries when queried from a variety of geographical vantage points and in various languages. We evaluate 15 LLMs from 13 locations in 12 different languages, producing over 700,000 responses.

### Research Artifacts

1. **Corpus (Artifact #1)**: 1,113 problematic statements across 5 categories, available in English and 12 other languages
2. **Testing Scripts (Artifact #2)**: Prompting and evaluation scripts (this directory)
3. **Results (Artifact #3)**: Over 700,000 LLM responses in CSV format
4. **DeBERTa Classifier (Artifact #4)**: Custom classifier to evaluate moderation strength. This is large and therefore hosted on huggingface: [https://huggingface.co/Tensorride/Classifier_30k](https://huggingface.co/Tensorride/Classifier_30k)

### Moderation Classification

Our pipeline classifies responses into three categories:
- **Unmoderated**: Direct answer to the problematic query
- **Hard Moderated**: Explicit refusal to respond
- **Soft Moderated**: Evasive or indirect responses

## Directory Structure

```
.
├── scripts/          # Artifact #2 - all testing and evaluation scripts
├── corpus/              # Artifact #1 - problematic statements corpus
│   ├── split/          # CSV files organized by country/category
│   ├── prompts.csv     # Prompt templates for different languages
│   └── prompts_with_location_info_*.csv  # Location-specific prompts
├── results/             # Artifact #3 - Full experiment results (700k+ responses)
│   ├── Offline/        # Results from locally-run models
│   └── Online/         # Results from API-based models
├── results_ae/          # Output directory for artifact evaluation runs
├── models/              # Model files for offline inference
│   └── AnonymousNDSSSubmitter12345--Classifier_30k/  # Artifact #4 - DeBERTa classifier (needs download from huggingface)
├── survey/              # Human survey data for translation validation
```

## Artifact Evaluation Guidance

For reviewers, we recommend the following evaluation approach:

### Quick Quality Check (Recommended)

1. **Corpus Sanity Check**: Review select statements from `corpus/split/` to verify content and formatting
2. **Results Quality Check**: Examine a few CSV files from `results/` to verify response format and quality
3. **DeBERTa Classifier Test**: 
   - Run `offline_soft_classifier.py` on a small subset of results (requires working setup running our custom DeBERTa classifier)
   - Manually verify classification accuracy for 10-20 responses
4. **Prompting Script Test**: 
   - Create an API key for one commercial LLM provider (e.g., OpenAI)
   - Run one of the `prompt_online_*.py` scripts on a few statements
   - Verify code correctness and response format
   - **Note**: This incurs a small charge (a few dollars at most for testing)

### In-Depth Evaluation (Optional)

Due to the large volume of data (700k+ statements), comprehensive manual review is not feasible. However, reviewers interested in deeper analysis can:
- Run the full classification pipeline on a subset of results
- Compare classifier outputs with manual annotations
- Test multiple prompting scripts with different models

## Folder Contents Details

### `corpus/` - Problematic Statements Corpus
Contains 1,113 problematic statements organized by:
- **Categories**: 5 types of problematic content
- **Countries**: Statements split by 13 different geographical vantage points
- **Languages**: Each statement available in English + 12 other languages
- **Format**: CSV files with columns for statement text, translations, metadata

### `results/` - Experiment Results
Contains over 700,000 LLM responses:
- **Offline/**: Responses from 7 locally-hosted models
- **Online/**: Responses from 8 API-based models
- **Organization**: Mirroring corpus structure (by country/category)
- **Format**: CSV files with prompt, response, timing, and classification columns

### `scripts/` - Testing and Evaluation Scripts
This directory contains:
- **Prompting scripts**: Query LLMs with corpus statements
- **Classification scripts**: Evaluate response moderation
- **All sensitive data removed**: API keys, usernames, absolute paths

### `models/` - Model Files
Contains downloaded model weights for:
- 7 offline LLMs for local inference
- DeBERTa classifier for moderation classification

### `survey/` - Human Validation Data
Human survey responses used to validate:
- Translation quality
- Geographic/cultural sensitivity

### `DeBERTa Training Set` - DeBERTa Training
Contains the training dataset and results used to develop the custom DeBERTa-based classifier:
- Training data with labeled examples
- Evaluation metrics

## Prerequisites

### Required Python Packages

- **For offline prompting scripts:**
  - `vllm` (for most offline models)
  - `torch`
  - `transformers`
  - `chardet`
  - `tqdm`

- **For online prompting scripts:**
  - `openai` (for ChatGPT, DeepSeek, XAI)
  - `anthropic` (for Claude)
  - `google-genai` (for Google GenAI)

- **For classification scripts:**
  - `pandas`
  - `torch`
  - `transformers`
  - `mistralai`
  - `openai`
  - `google-genai`
  - `tqdm`

### API Keys

Before running online scripts or classification scripts, you must set your API keys in the respective scripts:

- `prompt_online_chatgpt.py`: Set `OPENAI_API_KEY`
- `prompt_online_claude.py`: Set `CLAUDE_API_KEY`
- `prompt_online_deepseek.py`: Set `DEEPSEEK_API_KEY`
- `prompt_online_google.py`: Set `GOOGLE_API_KEY`
- `prompt_online_xai.py`: Set `XAI_API_KEY`
- `hard_classifier.py`: Set `OPENAI_API_KEY`, `MISTRAL_API_KEY`, `GOOGLE_API_KEY`
- `soft_classifier.py`: Set `OPENAI_API_KEY`, `GOOGLE_API_KEY`

### Models

For offline prompting, you need to download and place the following models in the `../models/` directory:

1. **CohereLabs--c4ai-command-a-03-2025** (for `prompt_offline_cohere.py`)
2. **deepseek-ai--DeepSeek-V3-0324** (for `prompt_offline_deepseek.py`)
3. **Qwen--Qwen2.5-72B-Instruct** (for `prompt_offline_qwen.py`)
4. **mistralai--Mistral-Small-3.1-24B-Instruct-2503** (for `prompt_offline_mistral.py`)
5. **TheBloke--WizardLM-30B-uncensored-AWQ** (for `prompt_offline_wizardlm.py`)
6. **meta-llama--Llama-3.3-70B-Instruct** (for `prompt_offline_llama.py`)
7. **google--gemma-3-27b-it** (for `prompt_offline_gemma.py`)
8. **AnonymousNDSSSubmitter12345--Classifier_30k** (for `offline_soft_classifier.py`)

## Scripts Overview

### Offline Prompting Scripts

These scripts run LLMs locally using the models in the `../models/` directory:

- **prompt_offline_cohere.py**: Run Cohere model inference
- **prompt_offline_deepseek.py**: Run DeepSeek model inference
- **prompt_offline_qwen.py**: Run Qwen model inference
- **prompt_offline_mistral.py**: Run Mistral model inference
- **prompt_offline_wizardlm.py**: Run WizardLM model inference
- **prompt_offline_llama.py**: Run Llama model inference
- **prompt_offline_gemma.py**: Run Gemma model inference

### Online Prompting Scripts

These scripts query LLM APIs (requires API keys):

- **prompt_online_chatgpt.py**: Query OpenAI GPT models
- **prompt_online_claude.py**: Query Anthropic Claude models
- **prompt_online_deepseek.py**: Query DeepSeek API
- **prompt_online_google.py**: Query Google GenAI API
- **prompt_online_xai.py**: Query XAI (Grok) API

### Classification Scripts

These scripts classify the LLM responses:

- **hard_classifier.py**: Binary classification (refusal vs. other) using multiple API models
- **soft_classifier.py**: Detailed moderation classification using OpenAI and Google APIs
- **offline_soft_classifier.py**: Classification using local DeBERTa model
- **soft_consensus.py**: Compute consensus labels from multiple classifiers

## Usage

### Running Offline Prompting

```bash
# Example: Run Qwen model
python prompt_offline_qwen.py
```

### Running Online Prompting

```bash
# Example: Run ChatGPT prompting
python prompt_online_chatgpt.py
```

### Running Classification

```bash
# Step 1: Run hard classification
python hard_classifier.py

# Step 2: Run soft classification (online models)
python soft_classifier.py

# Step 3: Run offline soft classification (DeBERTa)
python offline_soft_classifier.py

# Step 4: Compute consensus
python soft_consensus.py
```

## Output

All scripts write their results to CSV files in the `../results_ae/` directory, preserving the directory structure from the corpus.

## Citation

If you use these scripts in your research, please cite our paper:

Lipphardt, F., Ali, M., Banzer, M., Feldmann, A., & Gosain, D. There is No War in Ba Sing Se: A Global Analysis of Content Moderation in LLMs. In Proceedings of the Network and Distributed System Security (NDSS) Symposium 2026, San Diego, CA, USA, 23–27 February 2026. ISBN 979-8-9919276-8-0. https://dx.doi.org/10.14722/ndss.2026.240593
