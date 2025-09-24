# -------------------------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------------------------

import glob  # File pattern matching
import json  # JSON file handling
import os  # Operating system interfaces
import random  # Random number generation
import time  # Time-related functions
from pathlib import Path, PurePosixPath  # Filesystem path handling
from shutil import copyfile  # File copy operations
import subprocess  # Subprocess management
import sys  # System-specific parameters and functions


# -------------------------------------------------------------------------
# Virtual Environment Setup
# -------------------------------------------------------------------------

def create_virtual_environment(env_name=".venv"):
    """Create a virtual environment if it doesn't exist."""
    if not Path(env_name).exists():
        subprocess.run([sys.executable, "-m", "venv", env_name])
        print(f"Virtual environment '{env_name}' created.")
    else:
        print(f"Virtual environment '{env_name}' already exists.")

def activate_virtual_environment(env_name=".venv"):
    """Activate the virtual environment."""
    if os.name == 'nt':  # For Windows
        activate_script = Path(env_name) / "Scripts" / "activate"
    else:  # For Unix or MacOS
        activate_script = Path(env_name) / "bin" / "activate"
    if activate_script.exists():
        print(f"To activate the virtual environment, run: source {activate_script}")
    else:
        print("Activation script not found. Please ensure the virtual environment was created successfully.")

def install_packages(env_name=".venv"):
    """Install required packages into the virtual environment."""
    requirements = [
        'numpy',
        'pandas',
        'matplotlib',
        'nltk',
        'spacy',
        'gensim',
        'scikit-learn',
        'transformers',
        'torch',
        'pyTigerGraph',
        'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz'
    ]
    for package in requirements:
        subprocess.run([Path(env_name) / "bin" / "pip", "install", package])
        print(f"Installed {package}.")

# Create and activate the virtual environment
create_virtual_environment()
activate_virtual_environment()
install_packages()


# -------------------------------------------------------------------------
# Third-Party Library Imports
# -------------------------------------------------------------------------

# Data processing and linear algebra
import numpy as np  # Linear algebra operations
import pandas as pd  # Data manipulation and analysis

# Visualization
import matplotlib.pyplot as plt  # Plotting and data visualization
plt.style.use('ggplot')  # Apply ggplot style to plots

# Natural Language Processing (NLP)
import nltk  # Natural Language Toolkit for text processing
from nltk.corpus import stopwords  # Import stopwords corpus
from nltk.tokenize import sent_tokenize, word_tokenize  # Sentence and word tokenization

import spacy  # Advanced NLP library

from gensim import models  # Topic modeling and document similarity
from gensim.models import Phrases  # Phrase detection
from gensim import corpora  # Corpus handling for NLP

# Machine Learning and Deep Learning
from sklearn.metrics.pairwise import cosine_similarity  # Similarity metrics
from transformers import AutoTokenizer, AutoModel  # Pre-trained NLP models
import torch  # PyTorch for deep learning

# TigerGraph Database Connectivity (if needed)
import pyTigerGraph as tg  # TigerGraph database connector


# -------------------------------------------------------------------------
# Initialization and Setup
# -------------------------------------------------------------------------

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load scispaCy model
try:
    nlp = spacy.load('en_core_sci_lg')
except OSError:
    print("Failed to load scispaCy 'en_core_sci_lg' model.")