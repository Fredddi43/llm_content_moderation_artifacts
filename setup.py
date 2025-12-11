#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# Read long description from README if available
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="llm-content-moderation-artifacts",
    version="1.0.0",
    description="Scripts for evaluating LLM content moderation across geographical vantage points and languages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    packages=find_packages(),
    py_modules=[
        "hard_classifier",
        "soft_classifier",
        "soft_consensus",
        "offline_soft_classifier",
        "prompt_offline_cohere",
        "prompt_offline_deepseek",
        "prompt_offline_gemma",
        "prompt_offline_llama",
        "prompt_offline_mistral",
        "prompt_offline_qwen",
        "prompt_offline_wizardlm",
        "prompt_online_chatgpt",
        "prompt_online_claude",
        "prompt_online_deepseek",
        "prompt_online_google",
        "prompt_online_xai",
    ],
    install_requires=[
        # Core dependencies
        "pandas>=1.0.0",
        "tqdm>=4.60.0",
        
        # Offline prompting dependencies
        "vllm>=0.0.0",  # Version depends on specific setup
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "chardet>=4.0.0",
        
        # Online prompting dependencies
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "google-genai>=0.1.0",
        
        # Classification dependencies
        "mistralai>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

