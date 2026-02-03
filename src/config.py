"""
config.py - Central configuration for the Quantum Dataset Pipeline.
"""
import re
from pathlib import Path

class Config:
    # --- Paths ---
    BASE_DIR = Path(".")
    EXAM_ID = "66" 
    
    PAPER_LIST_PATH = BASE_DIR / f"paper_list_{EXAM_ID}.txt"
    OUTPUT_IMAGE_DIR = BASE_DIR / f"images_{EXAM_ID}"
    OUTPUT_JSON_PATH = BASE_DIR / f"dataset_{EXAM_ID}.json"
    OUTPUT_CSV_PATH = BASE_DIR / f"paper_list_counts_{EXAM_ID}.csv"
    PDF_CACHE_DIR = BASE_DIR / "raw_pdfs"

    # --- Pipeline Control ---
    MAX_TOTAL_IMAGES = 250
    
    # Network robustness
    REQUEST_TIMEOUT = 30       
    REQUEST_DELAY = 2.0        
    MAX_RETRIES = 3            
    BACKOFF_FACTOR = 2         

    # --- Heuristics: Regex Patterns ---
    CAPTION_PATTERN = re.compile(r"^(?:FIG(?:URE|\.)?|Tab(?:le|\.)?)\s*(\d+)", re.IGNORECASE)
    REF_PATTERN = re.compile(r'(?:in|see|shown in|as depicted in)\s+(?:FIG(?:URE)?\.?|Fig\.)\s*(\d+)', re.IGNORECASE)

    # ---  Bag of Words (Research Based) ---

    CIRCUIT_KEYWORDS = [
        "quantum circuit", "circuit diagram", "gate sequence", "pulse sequence", 
        "logic circuit", "schematic", "ansatz", "algorithm steps", 
        "wiring diagram", "computing the loss", "state generation",
      
        "quantum network", "teleportation protocol", "entanglement swapping",
        "error correction cycle", "syndrome extraction", "encoding circuit",
        "decoding circuit", "variational form", "unitary evolution",
        "gate decomposition", "transpilation", "logical qubit", "physical qubit"
    ]   

    # 2. Hardware/Physical Indicators (Low Confidence)
    # These often point to physical photos, plots, or CAD drawings.
    # We only want these IF they also contain gates (like boxes or lines).
    
    HARDWARE_KEYWORDS = [
        "transmon", "superconducting qubit", "ion trap", "cavity",
        "resonator", "interferometer", "josephson junction", "squid", 
        "hardware", "chip", "device", "processor", "layout", "setup",
        # New Additions:
        "dilution refrigerator", "cryostat", "waveguide", "coupler",
        "control line", "readout line", "qubit array", "ccd", "aom", "eom",
        "laser", "optical table", "fabricated", "micrograph"
    ]

    # 3. INTERNAL GATES (Visual Tokens found INSIDE images)
    
    IN_FIGURE_GATES = [
        " H ", " X ", " Y ", " Z ", " S ", " T ",
        " RX ", " RY ", " RZ ", "Oracle", "Ansatz", "Encoder", "Prep", "QFT", "InvQFT", "S",
        " CNOT ", " CZ ", " SWAP ", " iSWAP ", " CCNOT ", " CSWAP ",
        " Measure ", " M ", " |0> ", " |1> ", " |+> ", " |-> ",
        # Generic unitary/measurement boxes
        " U ", " U3 ",
        
        " U1 ", " U2 ", " CR ", " ECR ", " XX ", " YY ", " ZZ ",
        " Rxx ", " Ryy ", " Rzz ", " Toffoli ", " Fredkin ",
        " Barrier ", " Delay ", " Reset ", " Id ", " Identity ",
        " D ", " V ", " V† ", " SX ", " SX† "
    ]

    # 4. COMPONENT NAMES (For Metadata & Scoring)
    # Comprehensive list for semantic tagging.
    GATE_KEYWORDS = [
        # Single-qubit
        "Hadamard", "Pauli-X", "Pauli-Y", "Pauli-Z",
        "Phase Gate", "S Gate", "T Gate", "Rotation",
        "RX", "RY", "RZ", "Identity", "Reset",

        # Multi-qubit
        "CNOT", "controlled-NOT", "Controlled-NOT", "Toffoli", "CCNOT",
        "SWAP", "iSWAP", "CZ", "Controlled-Z", "Fredkin", "CSWAP",
        "Mølmer-Sørensen", "XX gate", "YY gate", "ZZ gate", "CR gate", "Cross-Resonance",

        # Measurement / state
        "Measurement", "Measure", "Basis state", "Bell state",
        "Ancilla", "Register", "|0>", "|1>", "|+>", "|->",
        "GHZ State", "W State", "Cluster State", "Graph State",

        # Hardware / components
        "Transmon", "Resonator", "Cavity", "Waveguide",
        "Josephson junction", "SQUID", "Beam Splitter", "BS",
        "Phase Shifter", "PS", "Detector", "Mirror", "Lens"
    ]

    # 5. VISUAL STOPWORDS (Kill Plots & Histograms)
    # Added specific plot metrics and axis labels.
    PLOT_AXIS_LABELS = [
        # Axes & units
        "frequency", "ghz", "mhz", "khz",
        "time (ns)", "time (us)", "time (s)", "time [ns]", "time [us]", "time [s]",
        "voltage", "mv", "flux", "current", "population",
        "fidelity", "infidelity", "error rate", "probability",
        "count rate", "kcps", "a.u.", "arbitrary units", "plot", "corner",
     
        "temperature (k)", "temperature (mk)", "energy (ev)", "wavelength (nm)",
        "detuning", "power (dbm)", "transmittance", "reflectance",
        "iterations", "epochs", "runtime", "latency", "overhead",
        "ratio", "scaling", "accuracy", "precision", "recall", "f1-score"
    ]

    # 6. DIAGRAM KILLERS (Generic Flowcharts / Networks / Non-circuit diagrams)
    # Added classical network and system architecture terms.
    BLOCK_DIAGRAM_TERMS = [
        # Flowchart / block-diagram language
        "flowchart", "block diagram", "block-diagram", "workflow",
        "pipeline", "data flow", "control flow", "process flow",

        # Generic conceptual diagrams / actors
        "layer", "stack", "support chain",
        "alice", "bob", "eve", "charlie",
        "environment", "total system", "user",
        "cloud", "server", "client", "internet", "channel",
        "attacker", "adversary", "deployment", "operational",
        "service", "extraction", "ruler", "corner", "voltage", "optimal", "result", "trade",
       
        "classical network", "classical control", "fpga", "dac", "adc",
        "mixing", "heterodyne", "homodyne", "electronics", "rack",
        "osi model", "tcp/ip", "neural network", "perceptron"
    ]

    # 7. ALGORITHMS (For 'quantum_problem' Metadata)
    # Major expansion to cover QEC, Verification, and Subroutines.
    ALGO_KEYWORDS = [
        # Standard algorithms
        "Shor", "Grover", "Deutsch-Jozsa", "Bernstein-Vazirani", "Simon",
        "QFT", "Quantum Fourier Transform", "QPE", "Phase Estimation",
        "Teleportation", "VQD", "GHZ", "W state", "Heisenberg",

        # NISQ / modern topics
        "VQE", "Variational Quantum Eigensolver",
        "QAOA", "Quantum Approximate Optimization",
        "QML", "Quantum Machine Learning",
        "Error Correction", "Surface Code", "Color Code",
        "Magic State Distillation", "Magic State",
        "Boson Sampling",
        "Toric Code",
        "State Preparation", "Tomography", "Compiler",
        
        "Amplitude Amplification", "Amplitude Estimation", "HHL", "Linear Systems",
        "Quantum Walk", "Randomized Benchmarking", "Gate Set Tomography",
        "Syndrome Extraction", "Stabilizer Measurement", "Repetition Code",
        "Bacon-Shor Code", "Steane Code", "Concatenated Code",
        "Transpilation", "Circuit Optimization", "Mapping", "Routing",
        "Variational Form", "Ansatz Preparation", "Cost Function Evaluation",
        "Kernel Method", "Support Vector Machine", "Classifier"
    ]

    # 8. STATE VISUALIZATION TERMS (Kill Bloch Spheres, etc.)
    # Added math-heavy visualization terms.
    STATE_VISUALIZATION_TERMS = [
        "q-bead", "q bead", "q-beads",
        "visualization", "visualisation",
        "visual representations",
        "color patch", "colour patch",
        "pixel matrix", "measurement statistics",
        "bit and corresponding qubit representations",

        # Bloch / sphere / geometry
        "bloch sphere", "bloch vector", "bloch ball",
        "sphere", "spherical", "crosssection of the bloch sphere",
        "polytope", "simplex", "tetrahedron",

        # Magic / state sets
        "stabilizer polytope", "magic state", "non-stabilizer",
        "state space", "state set",

        # General visualization words
        "representation of states",
        "diagram of states", "geometry of states",
        
        "wigner function", "quasiprobability", "phase space",
        "density matrix", "city plot", "hinton plot",
        "real part", "imaginary part", "complex plane"
    ]

    # 9. TOPOLOGY & GRAPH TERMS (Kill Static Lattices)
    #  Added graph theory and lattice geometry terms.
    TOPOLOGY_KEYWORDS = [
        "lattice", "grid", "surface code", "toric code", 
        "graph", "vertex", "plaquette", "stabilizer generator", 
        "cycle", "boundary", "quiver", "tensor network",
        # New Additions:
        "square lattice", "hexagonal lattice", "heavy-hex", "coupling map",
        "connectivity graph", "chimera graph", "pegasus graph",
        "edges", "nodes", "adjacency matrix", "dual lattice"
    ]
    
    # 10. STRONG CIRCUIT KEYWORDS (For "Keyword Immunity")
    # These override vetoes.
    STRONG_CIRCUIT_KEYWORDS = [
        "quantum circuit", "circuit diagram", "wiring diagram", 
        "gate sequence", "ansatz", "teleportation circuit"
    ]
    
    # --- Visual Filters ---
    MIN_IMAGE_WIDTH = 100
    MIN_IMAGE_HEIGHT = 100
    MIN_FILE_SIZE_BYTES = 2000 

    @classmethod
    def setup_directories(cls):
        cls.OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        cls.PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

Config.setup_directories()