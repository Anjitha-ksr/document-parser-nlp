"""
metadata_miner.py - Semantics Extraction Module.

This module is responsible for the "Intelligent" part of the pipeline.
It takes a raw visual `FigureCandidate` and mines it for semantic meaning,
using a combination of heuristics,
regex patterns, and context from the paper title.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Set
from config import Config
from figure_extraction import FigureCandidate


class MetadataMiner:
    """
    Extracts semantic metadata from figure candidates.

    This class aggregates information from multiple sources (Figure Text,
    Caption, Referencing Paragraph, and Paper Title) to produce a 
    rich metadata dictionary for the final dataset.
    """

    def mine(self, candidate: FigureCandidate, paper_title: str = "") -> Dict:
        """
        Extract semantic metadata and link it to source text positions.

        This method acts as the main entry point. It combines text sources,
        runs specific extractors for gates and algorithms, and compiles
        the final dictionary.

        Parameters
        ----------
        candidate : FigureCandidate
            The figure object containing image data, OCR text, and caption.
        paper_title : str, optional
            The title of the source paper. This is a high-confidence signal
            for algorithm detection (e.g., if the title mentions "Shor's Algorithm").
            Default is "".

        Returns
        -------
        Dict
            A standardized dictionary containing:
            - arxiv_number, page_number, figure_number
            - quantum_gates: List of unique gates found.
            - quantum_problem: The detected algorithm/problem.
            - descriptions: A list of descriptive strings (caption, context).
            - text_positions: Character spans for the descriptions.
        """
        # 1. Aggregate and CLEAN Text Sources
        # We apply cleaning immediately so all downstream logic works on standard text
        # This fixes issues with Mathematical Unicode (e.g. 𝐇 -> H, 𝜓 -> psi)
        caption = self._clean_text(candidate.caption_text)
        inside = self._clean_text(candidate.figure_content_text)
        context = self._clean_text(candidate.referencing_context)
        
        # Combine all text sources for a "broad sweep" search
        full_blob = f"{caption} {inside} {context}"

        # 2. Extract Gates
        # Uses a specialized two-tier strategy (Strict vs. Fuzzy)
        gates = self._extract_gates(inside, full_blob)

        # 3. Extract Problem/Algorithm
        # We pass the title explicitly as it's the strongest indicator
        problem = self._extract_problem(full_blob, paper_title)

        # 4. Descriptions & Positions
        descriptions: List[str] = []
        text_positions: List[Tuple[int, int]] = []

        # A. Always include the full clean caption (Primary description)
        if caption:
            descriptions.append(caption)
            span = getattr(candidate, "caption_span", (-1, -1))
            text_positions.append(span)

        # B. Include the referencing paragraph context (Secondary description)
        if context:
            descriptions.append(f"Context: {context}")
            span = getattr(candidate, "context_span", (-1, -1))
            text_positions.append(span)

        return {
            "arxiv_number": candidate.arxiv_id,
            "page_number": candidate.page_num,
            "figure_number": candidate.figure_idx + 1,
            "quantum_gates": sorted(list(gates)),
            "quantum_problem": problem,
            "descriptions": descriptions,
            "text_positions": text_positions, 
        }

    def _clean_text(self, text: str) -> str:
        """
        Aggressively cleans PDF artifacts and normalizes math.

        Fixes:
        1. Leader lines: ". . . . . . ." -> removed.
        2. Unicode Math: "U†", "|ψ⟩", "⊗" -> "U^dag", "|psi>", "(x)".
        3. Control chars: \u0000, \u0001 -> removed.
        """
        if not text:
            return ""

        # 1. Normalize Unicode (NFKC)
        # Fixes bold math fonts (𝐇 -> H) and compatibility characters
        text = unicodedata.normalize('NFKC', text)

        # 2. Specific Quantum/Math Replacements (Make it readable!)
        # We map common unicode math symbols to ASCII equivalents
        replacements = {
            "⟩": ">", "⟨": "<", 
            "†": "^dag", 
            "⊗": "(x)", "⊕": "(+)", 
            "×": "x", 
            "≡": "=", "≈": "~", 
            "−": "-", # unicode minus vs hyphen
            "ψ": "psi", "φ": "phi", "θ": "theta", "π": "pi",
            "α": "alpha", "β": "beta", "γ": "gamma",
            "√": "sqrt",
            "…": "...",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        # 3. Remove "Leader Lines" (The . . . . . artifacts)
        # Matches sequences of 3+ dots/dashes/underscores with optional spaces
        # e.g., ". . . .", "......", "_ _ _"
        text = re.sub(r'(?:[\.\-_]\s*){3,}', ' ', text)

        # 4. Remove Control Characters & Junk
        # Keep basic punctuation, letters, numbers, spaces
        text = "".join(c for c in text if c.isprintable())

        # 5. Collapse Whitespace (Fixes "  0/1   0/1")
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _extract_problem(self, text: str, title: str) -> str:
        """
        Identify the quantum algorithm or problem context.

        The logic follows a strict hierarchy of confidence:
        1. **Specific Keywords**: Checks if known algorithms (from Config) appear in the Title or Text.
        2. **Contextual Patterns**: Uses regex to find phrases like "Circuit for X".
        3. **Heuristics**: Looks for generic "X Problem" phrases.

        Parameters
        ----------
        text : str
            The combined text blob (caption + inside + context).
        title : str
            The paper title.

        Returns
        -------
        str
            The name of the algorithm (e.g., "Grover's Algorithm") or "Unknown".
        """
        # Combine title and text for a unified search space
        combined_text_lower = (title + " " + text).lower()
        
        # 1. Check Specific Known Algorithms (Longest First)
        # We sort by length to ensure "Variational Quantum Eigensolver" is matched
        # before just "Eigensolver" or "VQE".
        sorted_algos = sorted(Config.ALGO_KEYWORDS, key=len, reverse=True)
        for algo in sorted_algos:
            if algo.lower() in combined_text_lower:
                return algo

        # 2. Contextual Patterns (Smart Regex)
        # Looks for: "Circuit for [Something]", "Scheme for [Something]"
        # Captures up to 4 words after the keyword to get the full name.
        pattern = r"(?:circuit|scheme|protocol|network|algorithm)\s+for\s+((?:[\w\-]+\s?){1,4})"
        match = re.search(pattern, combined_text_lower)
        if match:
            # Clean up the result (e.g., "teleportation" or "generating ghz states")
            found = match.group(1).strip()
            # Filter out common junk words that might follow "for"
            if found not in ["the", "a", "an", "quantum"]: 
                return found.title()

        # 3. Heuristic: "Problem" suffix
        # e.g., "solving the hidden subgroup problem" -> "Hidden Subgroup Problem"
        match_prob = re.search(r"([\w\s]+) problem", combined_text_lower)
        if match_prob and len(match_prob.group(1)) < 30:
             return f"{match_prob.group(1).strip().title()} Problem"

        return "Unknown"

    def _extract_gates(self, inside_text: str, full_text: str) -> Set[str]:
        """
        Extract quantum gate names using a hybrid Two-Tier strategy.

        **Tier A: Strict Search (Single-Letter Gates)**
        - Targets: H, X, Y, Z, S, T, M
        - Source: ONLY `inside_text` (text physically inside the image).
        - Logic: Uses strict regex boundaries to prevent matching 'H' inside 'THE'.

        **Tier B: Fuzzy Search (Multi-Letter Gates)**
        - Targets: CNOT, Toffoli, SWAP, etc.
        - Source: `full_text` (Caption + Context + Inside).
        - Logic: Case-insensitive keyword matching.

        Parameters
        ----------
        inside_text : str
            Text found inside the figure bounding box.
        full_text : str
            The combined text blob of caption, context, and inside text.

        Returns
        -------
        Set[str]
            A set of unique gate names found.
        """
        found = set()
        
        # --- Tier A: Strict Single-Letter Gate Search ---
        # Only searches inside the figure to avoid false positives from prose.
        
        single_letter_map = {
            "H": "Hadamard", "X": "Pauli-X", "Y": "Pauli-Y", "Z": "Pauli-Z",
            "S": "Phase Gate", "T": "T Gate", "M": "Measurement"
        }
        
        # Regex explanation:
        # (^|[\s\(\[\]\.,])  -> Start of string OR separator (space, bracket, comma)
        # {char}             -> The target letter (e.g., 'H')
        # ($|[\s\)\]\.,])    -> End of string OR separator
        inside_upper = inside_text.upper()
        for char, full_name in single_letter_map.items():
            if re.search(rf"(^|[\s\(\[\]\.,]){char}($|[\s\)\]\.,])", inside_upper):
                found.add(full_name)

        # --- Tier B: Multi-letter Gates & Keywords ---
        # Searches full text context, allows case variations.
        
        text_lower = full_text.lower()
        
        # 1. Regex for standard keywords (CNOT, SWAP, Toffoli)
        for gate in Config.GATE_KEYWORDS:
            gate_clean = gate.lower()
            # Handle variations like "C-NOT" vs "CNOT"
            # escape(gate_clean) ensures special chars don't break regex
            pattern = r"\b" + re.escape(gate_clean).replace(r"\ ", r"[\s\-]*") + r"\b"
            if re.search(pattern, text_lower):
                found.add(gate)

        # 2. Special Symbols (Ket notation)
        # Often OCR'd as "|0>" or "| 0 >"
        if re.search(r"\|0\s*>", inside_text) or re.search(r"\|1\s*>", inside_text):
            found.add("Basis State")
        if re.search(r"\|\+\s*>", inside_text) or re.search(r"\|-\s*>", inside_text):
            found.add("Superposition State")

        return found