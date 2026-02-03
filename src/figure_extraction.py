"""
figure_extraction.py - Core Logic for Quantum Figure Detection.

This module is responsible for the geometric and semantic analysis of PDF pages.
It includes classes to:
1.  Represent a potential figure candidate (`FigureCandidate`).
2.  Filter out non-figures based on visual heuristics (`VisualFilter`).
3.  Classify candidates as 'Quantum Circuits' using keyword and density rules (`QuantumFigureClassifier`).
4.  Link captions to their corresponding visual elements (images/vectors) using spatial clustering (`FigureLinker`).
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from config import Config
from pdf_parsing import PDFPageParser, TextBlock


@dataclass
class FigureCandidate:
    """
    A data object representing a potential figure extracted from a PDF.

    This class acts as the central data structure passed between the
    linker, classifier, and metadata miner. It holds the image data,
    associated text, and classification results.

    Attributes
    ----------
    arxiv_id : str
        The unique identifier of the source paper.
    page_num : int
        The page number (1-indexed) where the figure was found.
    figure_idx : int
        The index of the figure on that specific page (0-indexed).
    bbox : Tuple[float, float, float, float]
        The bounding box of the figure image (x0, y0, x1, y1).
    image_bytes : bytes
        The raw PNG byte data of the rendered figure.
    caption_text : str
        The full text of the caption associated with this figure.
    caption_bbox : Tuple[float, float, float, float]
        The bounding box of the caption text block.
    caption_span : Tuple[int, int]
        The (start, end) character indices of the caption in the full page text.
        Default is (-1, -1).
    context_span : Tuple[int, int]
        The (start, end) character indices of the referencing paragraph.
        Default is (-1, -1).
    figure_content_text : str
        Text found physically inside the figure's bounding box.
    referencing_context : str
        The paragraph text in the paper body that cites this figure.
    bert_circuit_prob : float
        The probability score (0.0 to 1.0) assigned by the BERT model.
    source_type : str
        Indicates if the figure was built from 'vector_cluster' or 'raster' elements.
    is_circuit : bool
        The final binary decision: True if it is a quantum circuit, False otherwise.
    classification_score : float
        A heuristic score used for debugging/ranking confidence.
    debug_notes : List[str]
        A log of reasons why the figure was accepted or rejected.
    """
    arxiv_id: str
    page_num: int
    figure_idx: int
    bbox: Tuple[float, float, float, float]
    image_bytes: bytes
    caption_text: str
    caption_bbox: Tuple[float, float, float, float]

    # --- Provenance Tracking (Start, End indices in page text) ---
    caption_span: Tuple[int, int] = (-1, -1)
    context_span: Tuple[int, int] = (-1, -1)
    
    # Text Analysis Fields
    figure_content_text: str = ""
    referencing_context: str = ""
    bert_circuit_prob: float = 0.0  # BERT’s opinion: P(circuit | text)

    source_type: str = "vector_cluster"
    is_circuit: bool = False
    classification_score: float = 0.0
    debug_notes: List[str] = field(default_factory=list)


class VisualFilter:
    """
    A collection of static methods for fast, geometric filtering of candidates.
    
    These filters are applied early in the pipeline to discard obvious junk
    (like tiny icons or margin notes) before expensive operations like
    BERT inference or OCR.
    """

    @staticmethod
    def is_valid_candidate(
        width: int, height: int, file_size: int
    ) -> Tuple[bool, str]:
        """
        Perform a fast geometric check on the image dimensions.

        Runs BEFORE text extraction. Filters out tiny icons, long math
        equation strips, and tall narrow margin columns.

        Parameters
        ----------
        width : int
            Width of the candidate image in pixels.
        height : int
            Height of the candidate image in pixels.
        file_size : int
            Size of the image file in bytes.

        Returns
        -------
        Tuple[bool, str]
            (True, "Valid") if the image passes all checks.
            (False, Reason) if rejected.
        """
        # 1. Basic Dimensions & File Size
        if width < Config.MIN_IMAGE_WIDTH or height < Config.MIN_IMAGE_HEIGHT:
            return False, f"Too small dimensions ({width}x{height})"
        
        if file_size > 0 and file_size < Config.MIN_FILE_SIZE_BYTES:
             return False, f"File size too small ({file_size} bytes)"

        # 2. Aspect Ratio Filters (The "Strip & Snippet" Logic)
        if height == 0: 
            return False, "Height is zero"

        aspect_ratio = width / height

        # A. The "Equation Strip" Filter (Wide & Short)
        # Captures inline math like "H = ... + ..." which looks like a wide banner.
        if aspect_ratio > 5.0:
            return False, f"Rejected: Extreme aspect ratio (Wide Strip: {aspect_ratio:.1f})"

        # B. The "Margin Tower" Filter (Tall & Narrow)
        # Captures side notes, margin statistics, or vertical banners.
        if aspect_ratio < 0.2:
            return False, f"Rejected: Extreme aspect ratio (Tall Tower: {aspect_ratio:.1f})"

        return True, "Valid"

    @staticmethod
    def is_text_block_masquerading_as_image(candidate: FigureCandidate) -> bool:
        """
        Check if the candidate is actually just a list of text.

        Runs AFTER text extraction. Bullet points are a strong indicator
        that a 'figure' is actually just a text box or table caption
        mistakenly grouped as an image.

        Parameters
        ----------
        candidate : FigureCandidate
            The candidate to check.

        Returns
        -------
        bool
            True if it looks like a text block (Reject), False otherwise.
        """
        text_content = candidate.figure_content_text
        

        # 1. Bullet Point Check
    
        if any(marker in text_content for marker in ["•", "– ", "— ", "●"]):
            candidate.debug_notes.append("HardVeto:BulletPointsFound")
            return True
            
        return False


class QuantumFigureClassifier:
    """
    Advanced Hybrid Classifier for Quantum Circuits.

    This class determines if a figure is a 'Quantum Circuit' using a 
    multi-stage decision process:
    1.  **Hard Vetoes:** Immediately reject obvious non-circuits (plots, photos).
    2.  **Scoring:** Accumulate points based on keywords, visual tokens, and BERT probability.
    3.  **Final Decision:** Threshold logic based on the final score.

    Attributes
    ----------
    strong_circuit_keywords : List[str]
        Keywords that are highly specific to circuits (e.g., "teleportation circuit").
    hardware_veto_terms : List[str]
        Terms indicating a physical device photo (e.g., "micrograph").
    viz_veto_terms : List[str]
        Terms indicating a data plot (e.g., "histogram").
    math_veto_terms : List[str]
        Terms indicating a math equation or table.
    scale_units : List[str]
        Units found in microscopy images (e.g., "nm", "scale bar").
    plot_caption_phrases : List[str]
        Phrases common in plot captions (e.g., "versus", "as a function of").
    """

    def __init__(self) -> None:
        # --- 1. KEYWORD CONFIGURATION ---
        
        self.strong_circuit_keywords = [
            "quantum circuit", "circuit diagram", "wiring diagram", 
            "gate sequence", "ansatz", "teleportation circuit","QPE circuit"
        ]

        self.hardware_veto_terms = [
            "micrograph", "optical image", "sem image", "scanning electron",
            "fabricated", "fabrication", "lithography", "wafer", "substrate",
            "cryostat", "dilution", "resonator", "waveguide", "transmon",
            "graphene", "bilayer", "cavity", "setup", "apparatus",
            "schematic of the experiment", "experimental setup"
        ]

        # Added "probabilities" to catch bar charts
        self.viz_veto_terms = [
            "bloch sphere", "state visualization", "hilbert space", 
            "phase diagram", "energy level", "spectrum", "trajectory",
            "fidelity", "histogram", "distribution", "cumulative",
            "axis", "magnitude", "error rate", "runtime", "probabilities"
        ]

        self.math_veto_terms = [
            "representation of", "symbol", "list of", "table of", 
            "matrix representation", "equation", "theorem", "proof"
        ]
        
        self.scale_units = [r"μm", r"nm", r"mm", r"scale\s*bar"]

        self.plot_caption_phrases = [
            "as a function of", "versus", "vs.", "comparison of", "dependence of",
            "curve of", "evolution of", "distribution of", "histogram of", "fidelity of",
        ]

        self.plot_eval_terms = [
            "as a function of", "versus", "vs.", "variation of", "variance",
            "distribution", "histogram", "spectrum", "probability distribution",
            "performance", "accuracy", "loss", "evaluation", "benchmark",
            "simulation", "simulated", "numerical", "experiment", "experimental",
            "measurement", "measured", "data points", "fit", "fitting",
            "ratio", "boundary", "region", "curve", "contour",
        ]

        self.plot_strong_terms = [
            "simulation results", "numerical simulation", "measurement results",
            "experimental results", "evaluation of", "performance evaluation",
        ]

    def is_hard_vetoed(self, candidate: FigureCandidate) -> bool:
        """
        Run ONLY the fast Phase 1 rule-based vetoes.

        This method is used as a pre-filter before running expensive models
        like BERT. If it returns True, the candidate is definitely garbage.

        Parameters
        ----------
        candidate : FigureCandidate
            The candidate to check.

        Returns
        -------
        bool
            True if the candidate is rejected, False if it survives.
        """
        # Bullet points
        if VisualFilter.is_text_block_masquerading_as_image(candidate):
            return True

        caption = candidate.caption_text.lower()
        inside = candidate.figure_content_text.lower()
        context = candidate.referencing_context.lower()
        full_text_blob = f"{caption} {inside} {context}"

        has_strong_kw = any(kw in full_text_blob for kw in self.strong_circuit_keywords)

        # 1. Hardware Veto
        if any(term in inside or term in caption for term in self.hardware_veto_terms):
            return True
            
        for unit in self.scale_units:
            if re.search(rf"\b\d+\s*{unit}\b", inside):
                return True

        # 2. Density Vetoes (Only if no strong keyword)
        if not has_strong_kw:
            if self._is_numeric_dense(inside):
                return True
            if self._is_prose(inside):
                return True
            if self._is_math_dense(inside):
                return True

        # 3. Viz Veto
        if any(term in full_text_blob for term in self.viz_veto_terms):
            if any(term in caption for term in self.viz_veto_terms):
                return True

        # 4. Math Definition Veto
        if any(term in caption for term in self.math_veto_terms):
            return True
            
        # 5. Internal Plot Keywords
        plot_keywords_inside = [
            "gurobi", "cplex", "solver", "optimizer", 
            "savings", "relative", "proportion", "count", "frequency", 
            "cpu time", "seconds", "iterations", "epochs", 
            "legend", "run", "instance", "p (mev)", "δ (deg)", "error rate"
        ]
        if any(kw in inside for kw in plot_keywords_inside):
            return True
            
        if "lattice" in full_text_blob and "circuit" not in full_text_blob:
            return True
            
        return False

    def classify(self, candidate: FigureCandidate) -> bool:
        """
        Perform full classification of a surviving candidate.

        This method calculates a confidence score based on keyword presence,
        BERT probability, and text context penalties.

        Parameters
        ----------
        candidate : FigureCandidate
            The candidate to classify. The method modifies the candidate's
            `is_circuit`, `classification_score`, and `debug_notes` fields.

        Returns
        -------
        bool
            True if it is a quantum circuit, False otherwise.
        """
        candidate.debug_notes.clear()

        # Bullet points are still a dead giveaway for text lists
        if VisualFilter.is_text_block_masquerading_as_image(candidate):
            candidate.is_circuit = False
            return False

        caption = candidate.caption_text.lower()
        inside = candidate.figure_content_text.lower()
        context = candidate.referencing_context.lower()
        full_text_blob = f"{caption} {inside} {context}"
        bert_p = getattr(candidate, "bert_circuit_prob", 0.0)

        # Check for Strong Keywords early (for Immunity)
        has_strong_kw = any(kw in full_text_blob for kw in self.strong_circuit_keywords)

        # --- PHASE 1: HARD VETOES ---

        # 1. Hardware Veto (ALWAYS APPLIES - "Schematic of setup" != Circuit)
        if any(term in inside or term in caption for term in self.hardware_veto_terms):
            candidate.debug_notes.append("HardVeto:HardwareTerms")
            candidate.is_circuit = False
            return False
            
        for unit in self.scale_units:
            if re.search(rf"\b\d+\s*{unit}\b", inside):
                candidate.debug_notes.append(f"HardVeto:ScaleUnit({unit})")
                candidate.is_circuit = False
                return False

        # 2. Density Vetoes (APPLY ONLY IF NO STRONG KEYWORD)
        # This saves "Truth Tables" and "Code Blocks" that are labeled "Circuit Diagram"
        if not has_strong_kw:
            if self._is_numeric_dense(inside):
                candidate.debug_notes.append("HardVeto:NumericDensity(Plot/Table)")
                candidate.is_circuit = False
                return False

            if self._is_prose(inside):
                candidate.debug_notes.append("HardVeto:ProseDensity")
                candidate.is_circuit = False
                return False
                
            if self._is_math_dense(inside):
                candidate.debug_notes.append("HardVeto:MathSymbolDensity")
                candidate.is_circuit = False
                return False

        # 3. Visualization/Plot Keywords Veto
        if any(term in full_text_blob for term in self.viz_veto_terms):
            if any(term in caption for term in self.viz_veto_terms):
                candidate.debug_notes.append("HardVeto:Viz/PlotKeyword")
                candidate.is_circuit = False
                return False

        # 4. Math Definition Veto
        if any(term in caption for term in self.math_veto_terms):
            candidate.debug_notes.append("HardVeto:Definition/Math")
            candidate.is_circuit = False
            return False
            
        # 5. Internal Plot Keywords
        plot_keywords_inside = [
            "gurobi", "cplex", "solver", "optimizer", 
            "savings", "relative", "proportion", "count", "frequency", 
            "cpu time", "seconds", "iterations", "epochs", 
            "legend", "run", "instance", "p (mev)", "δ (deg)", "error rate"
        ]
        if any(kw in inside for kw in plot_keywords_inside):
            candidate.debug_notes.append("HardVeto:PlotKeywordsInside")
            candidate.is_circuit = False
            return False
            
        if "lattice" in full_text_blob and "circuit" not in full_text_blob:
            candidate.debug_notes.append("HardVeto:LatticeNoCircuit")
            candidate.is_circuit = False
            return False


        # --- PHASE 2: SCORING ---
        score = 0
        reasons = []

        inside_gate_hits = self._count_strict_gates(candidate.figure_content_text)
        
        if inside_gate_hits > 0:
            score += min(inside_gate_hits, 3)
            reasons.append(f"GatesIn:{inside_gate_hits}")

        if has_strong_kw:
            score += 3
            reasons.append("StrongCircuitKW")
        elif "schematic" in full_text_blob or "diagram" in full_text_blob:
            score += 1
            reasons.append("WeakDiagramKW")
            
        if any(term in caption for term in ["fig", "figure"]):
             pass 

        if bert_p >= 0.8:
            score += 4
            reasons.append(f"BERTStrong:{bert_p:.2f}")
        elif bert_p >= 0.6:
            score += 2
            reasons.append(f"BERTMedium:{bert_p:.2f}")
        elif bert_p >= 0.4:
            score += 1
            reasons.append(f"BERTWeak:{bert_p:.2f}")
        elif bert_p < 0.2:
            score -= 2
            reasons.append(f"BERTNegative:{bert_p:.2f}")

        # Context Penalties
        viz_hits = sum(1 for w in Config.STATE_VISUALIZATION_TERMS if w in caption or w in context)
        if viz_hits > 0:
            score -= 1
            reasons.append(f"StateViz:{viz_hits}")

        if any(phrase in caption for phrase in self.plot_caption_phrases):
            score -= 3
            reasons.append("PlotCaption")

        if any(term in full_text_blob for term in ["histogram", "distribution", "spectrum", "fidelity"]):
            score -= 2
            reasons.append("PlotKW")

        if any(term in full_text_blob for term in self.plot_eval_terms):
            score -= 2
            reasons.append("PlotEvalKW")
            
        if any(term in full_text_blob for term in self.plot_strong_terms):
            score -= 3
            reasons.append("PlotEvalStrong")
            
        if "qubit" in full_text_blob and inside_gate_hits == 0:
            score -= 2
            reasons.append("QubitsNoGates")
        
        if any(term in full_text_blob for term in ["lattice", "topological code", "stabilizer"]):
            score -= 8
            reasons.append("LatticeDiagram")


        # --- PHASE 3: FINAL DECISION ---
        is_valid = False
        
        # Case A: Strong Score (>=3)
        if score >= 3:
            # Paper Tiger Check: Don't accept on KW alone if BERT hates it
            if inside_gate_hits == 0 and bert_p < 0.25:
                 reasons.append("HighKeywordScore_But_VisualsFail")
                 is_valid = False
            else:
                 is_valid = True
            
        # Case B: Mid Score (>=2)
        elif score >= 2:
            if bert_p > 0.3:
                is_valid = True
            else:
                reasons.append("Score2_But_BERT_Low")
                
        # Case C: Lonely Gate (UPDATED: Loose Threshold)
        elif inside_gate_hits > 0:
            # If we found a gate pattern, we trust it unless BERT is totally negative.
            # Lowered from 0.60 to 0.35 to improve recall.
            if bert_p > 0.35: 
                is_valid = True
                reasons.append("LonelyGate_ConfirmedByBERT")
            else:
                reasons.append("LonelyGate_RejectedByBERT")
        
        # Case D: BERT Rescue
        elif bert_p > 0.92:
            is_valid = True
            reasons.append("BERT_Rescue")

        candidate.classification_score = score
        candidate.inside_gate_hits = inside_gate_hits
        candidate.debug_notes.append(f"Score={score}, gates_in={inside_gate_hits}, reasons={reasons}")
        candidate.is_circuit = is_valid
        candidate.debug_notes.append("Accepted as circuit." if is_valid else "Rejected as non-circuit.")
        
        return is_valid

    # --- HELPER METHODS ---

    def _count_strict_gates(self, text: str) -> int:
        """Counts regex matches for gates like CNOT, H, |0>."""
        gate_patterns = [
            r"\bCNOT\b", r"\bSWAP\b", r"\bMeasure\b", r"\bToffoli\b",
            r"\b[HXYZ]\b", r"\b[R][xyz]\b", r"\|0\⟩", r"\|1\⟩", r"\|\+\⟩",
        ]
        hits = 0
        for pat in gate_patterns:
            flags = 0 if len(pat) < 10 else re.IGNORECASE
            if re.search(pat, text, flags=flags):
                hits += 1
        return hits

    def _is_numeric_dense(self, text: str) -> bool:
        """Returns True if text is >15% numbers (indicates a table/plot axis)."""
        tokens = text.split()
        if len(tokens) < 5: return False
        num_pattern = r'^-?\d+(\.\d+)?(e-?\d+)?$'
        num_count = sum(1 for t in tokens if re.match(num_pattern, t.strip(".,()[]")))
        return (num_count / len(tokens)) > 0.15

    def _is_math_dense(self, text: str) -> bool:
        """Returns True if text contains many mathematical symbols (indicates equation)."""
        math_symbols = set("= + - / * ∑ ∫ ∂ √ π θ σ λ φ Δ Ω ∞ ≈ ≠ ≤ ≥ < >".split())
        symbol_hits = sum(1 for char in text if char in math_symbols)
        if len(text) < 50 and symbol_hits > 2: return True
        if len(text) >= 50 and (symbol_hits / len(text)) > 0.05: return True
        return False

    def _is_prose(self, text: str) -> bool:
        """Returns True if text is >25% stopwords (indicates a paragraph)."""
        stop_words = {"the", "and", "of", "in", "to", "a", "is", "that", "for", "with", "by"}
        tokens = text.split()
        if len(tokens) < 10: return False
        stop_hits = sum(1 for w in tokens if w.lower() in stop_words)
        return (stop_hits / len(tokens)) > 0.25
    
class FigureLinker:
    """
    Connects textual captions to their corresponding visual elements.

    This class implements a 'Sweep and Merge' algorithm to find all vector
    paths and raster images that belong to a specific caption.
    """

    def __init__(self, parser: PDFPageParser) -> None:
        self.parser = parser

    def extract_figures_from_page(
        self, arxiv_id: str, page_num: int
    ) -> List[FigureCandidate]:
        """
        Main entry point for extracting figures from a single page.

        1. Identifies all captions using regex.
        2. For each caption, scans the area above it for visual elements.
        3. Merges those elements into a single bounding box.
        4. Renders the clip to an image.
        5. Extracts text context.

        Parameters
        ----------
        arxiv_id : str
            The paper ID.
        page_num : int
            The page index (0-based).

        Returns
        -------
        List[FigureCandidate]
            A list of extracted candidates.
        """
        page_data = self.parser.parse_page(page_num)
        text_blocks = page_data["text_blocks"]
        vector_clusters = page_data["vector_clusters"]
        images = page_data["images"]
        full_page_text = page_data.get("page_text", "")

        page_width = page_data.get("width", 612)
        if not page_data.get("width") and text_blocks:
            page_width = max(b.bbox[2] for b in text_blocks) + 50

        # Find all text blocks matching "Figure X" or "Fig X"
        captions = [
            block for block in text_blocks
            if Config.CAPTION_PATTERN.search(block.text)
        ]

        candidates: List[FigureCandidate] = []

        for idx, cap_block in enumerate(captions):
            # The core logic: find components above this caption
            merged_bbox, source_type = self._find_and_merge_figure_components(
                cap_block, vector_clusters, images, page_width
            )

            if not merged_bbox:
                continue

            x0, y0, x1, y1 = merged_bbox
            # Add padding to the bounding box to capture axes labels or wire ends
            final_bbox = (
                max(0.0, x0 - 45.0),
                max(0.0, y0 - 15.0),
                min(page_width, x1 + 10.0),
                y1 + 10.0,
            )
            
            # --- OPTIMIZATION: Lazy Rendering ---
            w = final_bbox[2] - final_bbox[0]
            h = final_bbox[3] - final_bbox[1]
            
            # 1. Check Geometry BEFORE Rendering (Saves time on tiny/huge clips)
            if w < Config.MIN_IMAGE_WIDTH or h < Config.MIN_IMAGE_HEIGHT:
                continue

            # 2. Render only if dimensions are promising
            final_bytes = self.parser.render_page_clip(page_num, final_bbox)
            
            # 3. Check File Size (requires bytes)
            if len(final_bytes) < Config.MIN_FILE_SIZE_BYTES:
                continue
            # ------------------------------------

            content_text = self._scan_inside_figure(text_blocks, final_bbox)

            match = Config.CAPTION_PATTERN.search(cap_block.text)
            fig_num_str = match.group(1) if match else str(idx + 1)
            
            # --- UPDATED: Extract Context WITH SPAN ---
            context_text, context_span = self._extract_referencing_context_with_span(
                full_page_text, fig_num_str
            )
            
            # --- NEW: Find Caption Span in Full Text ---
            cap_start = full_page_text.find(cap_block.text)
            cap_span = (-1, -1)
            if cap_start != -1:
                cap_span = (cap_start, cap_start + len(cap_block.text))
            # -------------------------------------------

            candidates.append(
                FigureCandidate(
                    arxiv_id=arxiv_id,
                    page_num=page_num + 1,
                    figure_idx=idx,
                    bbox=final_bbox,
                    image_bytes=final_bytes,
                    caption_text=cap_block.text,
                    caption_bbox=cap_block.bbox,
                    caption_span=cap_span,         
                    figure_content_text=content_text,
                    referencing_context=context_text,
                    context_span=context_span,     
                    source_type=source_type,
                )
            )

        return candidates

    def _find_and_merge_figure_components(
        self,
        caption: TextBlock,
        clusters: List[Tuple],
        images: object,
        page_width: float,
    ) -> Tuple[Optional[Tuple], str]:
        """
        Identify visual components spatially associated with a caption.

        This handles two column layouts by checking the width of the caption.
        If the caption is wide (spanning page), it searches the whole width.
        If narrow, it only searches the column above the caption.

        Parameters
        ----------
        caption : TextBlock
            The caption block.
        clusters : List[Tuple]
            List of vector clusters (x0, y0, x1, y1).
        images : List[object]
            List of raster images (MuPDF image objects).
        page_width : float
            Total width of the page.

        Returns
        -------
        Tuple[Optional[Tuple], str]
            (merged_bbox, source_type). Returns (None, "none") if nothing found.
        """
        cap_bbox = caption.bbox
        cap_center_x = (cap_bbox[0] + cap_bbox[2]) / 2
        cap_width = cap_bbox[2] - cap_bbox[0]

        is_wide = cap_width > (page_width * 0.6)
        search_x_min, search_x_max = 0.0, page_width

        # Determine column boundaries based on caption position
        if not is_wide:
            if cap_center_x < page_width / 2:
                search_x_max = page_width * 0.65  # Left column
            else:
                search_x_min = page_width * 0.35  # Right column

        def collect_components(x_min, x_max):
            comps = []
            # Gather vector clusters above the caption within column bounds
            for clust in clusters:
                if clust[3] <= cap_bbox[1] and clust[2] > x_min and clust[0] < x_max:
                    comps.append(
                        {"bbox": list(clust), "bottom": clust[3], "type": "vector"}
                    )

            # Gather raster images above the caption within column bounds
            for img in images:
                if img.bbox[3] <= cap_bbox[1] and img.bbox[2] > x_min and img.bbox[0] < x_max:
                    comps.append(
                        {"bbox": list(img.bbox), "bottom": img.bbox[3], "type": "raster"}
                    )
            return comps

        valid_components = collect_components(search_x_min, search_x_max)
        
        # Fallback: If nothing found in column, try full width (misaligned caption case)
        if not valid_components and not is_wide:
            valid_components = collect_components(0.0, page_width)

        if not valid_components:
            return None, "none"

        # Sort components bottom-up (closest to caption first)
        valid_components.sort(key=lambda x: x["bottom"], reverse=True)
        merged_bbox = valid_components[0]["bbox"]
        primary_type = valid_components[0]["type"]

        # Merge upwards: Keep adding components as long as the vertical gap is small.
        # This groups composite sub-figures (e.g., Fig 1a and 1b).
        current_top = merged_bbox[1]
        for comp in valid_components[1:]:
            if current_top - comp["bottom"] < 150.0:
                merged_bbox[0] = min(merged_bbox[0], comp["bbox"][0])
                merged_bbox[1] = min(merged_bbox[1], comp["bbox"][1])
                merged_bbox[2] = max(merged_bbox[2], comp["bbox"][2])
                merged_bbox[3] = max(merged_bbox[3], comp["bbox"][3])
                current_top = merged_bbox[1]
            else:
                break

        return tuple(merged_bbox), primary_type

    def _extract_referencing_context_with_span(self, page_text: str, fig_num: str) -> Tuple[str, Tuple[int, int]]:
        """
        Find the text passage in the main body that refers to the figure.
        
        Searches for "Fig. X" or "Figure X" and extracts surrounding text.
        Also returns the character span indices.
        """
        patterns = [f"Fig\\. {fig_num}", f"Figure {fig_num}", f"FIG\\. {fig_num}"]
        for pat in patterns:
            for match in re.finditer(pat, page_text):
                # Grab 250 characters before and after the mention
                start = max(0, match.start() - 250)
                end = min(len(page_text), match.end() + 250)
                
                # Extract and clean text
                raw_text = page_text[start:end]
                clean_text = raw_text.replace("\n", " ")
                
                return clean_text, (start, end)
                
        return "", (-1, -1)

    def _scan_inside_figure(
        self, blocks: List[TextBlock], bbox: Tuple
    ) -> str:
        """
        Extract text found physically inside the figure's bounding box.
        Useful for reading axis labels or gate labels (e.g., "|0>", "H").
        """
        texts = []
        x0, y0, x1, y1 = bbox
        for b in blocks:
            bx0, by0, bx1, by1 = b.bbox
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            # Check if text block center is inside figure bbox
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                texts.append(b.text)
        return " ".join(texts)