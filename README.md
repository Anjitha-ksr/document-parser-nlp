# Quantum Image Dataset Builder

## Overview
The **Quantum Image Dataset Builder** is an automated pipeline designed to create high-quality datasets of quantum circuit diagrams from scientific literature (arXiv). 

It downloads PDFs, extracts visual figures, filters out non-circuit images (like plots and photos) using a hybrid **Heuristic + BERT** classification system, and mines semantic metadata (gate types, algorithm names) from the surrounding text.

## Features
* **Multiprocessing Pipeline:** Efficiently processes multiple papers in parallel.
* **Robust PDF Parsing:** Uses `PyMuPDF` with **IAAA Clustering** to correctly reconstruct vector-based circuit diagrams.
* **Hybrid Classification:**
    * **Heuristic Filter:** Rejects plots, tables, and prose based on visual density and keywords.
    * **BERT Model:** A fine-tuned DistilBERT model (`bert_circuit_new_model`) validates candidates to prevent false positives.
* **Semantic Mining:** Extracts specific quantum gates (e.g., CNOT, H, RX) and algorithm names (e.g., "Shor's Algorithm") using text-positional analysis.
* **Provenance Tracking:** Every extracted data point includes the source text span (character indices) for verification.

## Project Structure

| File | Description |
| :--- | :--- |
| `main.py` | **Entry Point**. Orchestrates the multiprocessing workers, manages the job queue, and saves results. |
| `config.py` | Central configuration (paths, regex patterns, keyword lists, and the `EXAM_ID`). |
| `figure_extraction.py` | Handles geometric analysis, candidate creation, and the **QuantumFigureClassifier** logic. |
| `metadata_miner.py` | Extracts semantic meaning (gates, algorithms) and cleans text artifacts. |
| `pdf_parsing.py` | Low-level wrapper for PyMuPDF; handles text extraction and vector clustering. |
| `bert_helper.py` | Wrapper for the `transformers` library to load and run the BERT classification model. |
| `data_management.py` | Handles robust arXiv downloading (retries, caching) and CSV state tracking. |
| `train_bert.py` | (Optional) Script to fine-tune the BERT model on a labeled CSV dataset. |

## Installation

1.  **Clone the repository** (or unzip the project files).
2.  **Install Dependencies**:
    Ensure you have Python 3.8+ installed. Run the following command:
    ```bash
    pip install -r requirements.txt
    ```

## Setup & Configuration

### 1. The Paper List
The pipeline expects a text file containing the list of arXiv IDs to process. 
* **File Name:** `paper_list_66.txt` (matches the `EXAM_ID="66"` in `config.py`).
* **Format:** One arXiv ID per line (e.g., `2101.00001`).

### 2. The BERT Model
The pipeline requires a pre-trained or fine-tuned BERT model to exist in the local directory.
* **Directory Name:** `bert_circuit_new_model/`
* If this folder is missing, you must run `train_bert.py` first (assuming you have the training data `labels_merged.csv`) or download the model weights.

## Usage

To start the extraction pipeline, simply run:

```bash
python main.py
