# bert_helper.py
"""
bert_helper.py - BERT-based Text Classification Utility.

This module provides a wrapper class for loading a fine-tuned BERT model
and performing batch inference on figure candidates. It is used to
classify whether a figure is a quantum circuit based on its caption
and referencing context text.
"""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from figure_extraction import FigureCandidate


class CircuitTextBERT:
    """
    A helper class to manage the BERT model for text classification.

    This class handles the initialization of the tokenizer and model,
    manages device placement (CPU vs GPU), and performs efficient
    batch inference to annotate FigureCandidate objects with a
    probability score.

    Attributes
    ----------
    device : torch.device
        The computation device (CUDA or CPU) where the model is loaded.
    tokenizer : AutoTokenizer
        The loaded BERT tokenizer for text preprocessing.
    model : AutoModelForSequenceClassification
        The loaded BERT model for sequence classification.
    """

    def __init__(self, model_dir: str = "bert_circuit_new_model") -> None:
        """
        Initialize the CircuitTextBERT wrapper.

        Loads the pre-trained/fine-tuned model and tokenizer from the
        specified directory and moves the model to the appropriate
        computation device (GPU if available).

        Parameters
        ----------
        model_dir : str, optional
            Path to the directory containing the saved model and tokenizer.
            Default is "bert_circuit_new_model".
        """
        # Determine the fastest available device (CUDA GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the tokenizer and model from the local directory
        # The tokenizer converts text into numerical token IDs
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load the classification model architecture and weights
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        # Move the model to the selected device and set to evaluation mode
        # (This disables dropout and other training-specific layers)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def annotate(self, candidates: List[FigureCandidate]) -> None:
        """
        Perform batch inference on a list of figure candidates.

        This method extracts text from each candidate, runs the BERT model,
        and attaches the resulting circuit probability score directly to
        the candidate object.

        Parameters
        ----------
        candidates : List[FigureCandidate]
            A list of FigureCandidate objects to be analyzed.
            The list is modified in-place: the 'bert_circuit_prob' attribute
            is added to each candidate.

        Returns
        -------
        None
            The operation modifies the input objects directly.
        """
        # Optimization: Return immediately if the list is empty to avoid processing overhead
        if not candidates:
            return

        # Pre-process text: Combine caption and context for maximum semantic signal.
        # Replaces newlines with spaces to ensure a continuous text stream for BERT.
        texts = [
            f"{c.caption_text} {c.referencing_context}".replace("\n", " ")
            for c in candidates
        ]

        # Tokenize the batch of texts.
        # padding=True: Pad all sequences to the length of the longest in the batch.
        # truncation=True: Truncate sequences longer than max_length.
        # return_tensors="pt": Return PyTorch tensors.
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        # Move all input tensors (input_ids, attention_mask, etc.) to the GPU/CPU
        enc = {k: v.to(self.device) for k, v in enc.items()}

        # Perform the forward pass through the model to get raw logits.
        # **enc unpacks the dictionary arguments into the model call.
        outputs = self.model(**enc)
        
        # Apply Softmax to convert logits into probabilities (0.0 to 1.0).
        # We take index [:, 1] assuming class 1 represents "is_circuit".
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().tolist()

        # Map the calculated probabilities back to the original candidate objects
        for cand, p in zip(candidates, probs):
            cand.bert_circuit_prob = float(p)