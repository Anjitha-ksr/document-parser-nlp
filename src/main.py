"""
main.py - The Pipeline Orchestrator (QuantumImageDatasetBuilder).

This module serves as the entry point and controller for the entire
dataset generation pipeline. It orchestrates the following concurrent operations:
1.  **Job Management**: Fetches arXiv IDs from the queue.
2.  **Resource Allocation**: Manages a pool of worker processes.
3.  **Data Flow**: Distributes PDFs to workers and aggregates results.
4.  **Persistence**: Periodically flushes metadata to JSON and progress to CSV.

Refactored for Multiprocessing with Robust Queue Management to ensure high throughput.
"""

import json
import time
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

from config import Config
from data_management import PaperListManager, ArxivDownloader
from pdf_parsing import PDFPageParser
from bert_helper import CircuitTextBERT
from figure_extraction import FigureCandidate, VisualFilter, QuantumFigureClassifier, FigureLinker
from metadata_miner import MetadataMiner

# --- Global Worker State ---
# This variable exists only in worker processes to hold the heavy BERT model.
# It allows us to load the model once per process, rather than once per paper.
_worker_bert_helper: Optional[CircuitTextBERT] = None

def worker_init() -> None:
    """
    Initializer function for worker processes.
    
    This function is called once when each worker process starts. It loads
    the heavy BERT model into the global scope of the worker, preventing
    repeated model loading/unloading overhead.
    """
    global _worker_bert_helper
    try:
        # print(f"[Worker {os.getpid()}] Initializing BERT model...")
        _worker_bert_helper = CircuitTextBERT()
    except Exception as e:
        print(f"[Worker {os.getpid()}] Failed to initialize BERT: {e}")

def process_paper_task(arxiv_id: str, pdf_path: Path) -> Tuple[str, int, List[Tuple[str, Dict]], List[str]]:
    """
    Worker function to process a single PDF file.

    This function encapsulates the entire extraction logic for one paper:
    parsing, candidate extraction, classification, and metadata mining.
    It runs isolated inside a worker process.

    Parameters
    ----------
    arxiv_id : str
        The unique arXiv identifier for the paper.
    pdf_path : Path
        Filesystem path to the downloaded PDF.

    Returns
    -------
    Tuple[str, int, List[Tuple[str, Dict]], List[str]]
        A tuple containing:
        1. arxiv_id: The ID of the processed paper.
        2. valid_count: Number of valid circuits found.
        3. mined_results: List of (filename, metadata_dict) tuples.
        4. debug_logs: List of error/warning strings from the worker.
    """
    # 1. Setup local lightweight tools
    # These classes are cheap to instantiate and don't need to be shared.
    parser = PDFPageParser(pdf_path)
    linker = FigureLinker(parser)
    classifier = QuantumFigureClassifier()
    miner = MetadataMiner()
    
    valid_count = 0
    mined_results = [] # Stores (filename, metadata)
    debug_logs = []

    # --- NEW: Extract Paper Title for Better Metadata Mining ---
    # We attempt to fetch the title from PDF metadata or the first page text
    # to provide better context for the MetadataMiner.
    paper_title = ""
    try:
        # 1. Try Metadata (fastest/cleanest)
        # MuPDF docs often have metadata dictionary
        if hasattr(parser.doc, 'metadata') and parser.doc.metadata and 'title' in parser.doc.metadata:
            candidate_title = parser.doc.metadata['title']
            # Basic validation to avoid default/empty titles
            if candidate_title and candidate_title.strip() and "replace with" not in candidate_title.lower():
                paper_title = candidate_title.strip()

        # 2. Fallback: First page text heuristic (common for arXiv)
        if not paper_title:
            # Get first block of text from first page
            page_zero_text = parser.doc[0].get_text("text").split('\n')
            for line in page_zero_text[:6]: # Scan first few lines
                clean = line.strip()
                # Simple filter to skip "arXiv:..." or empty lines to find the likely title
                if clean and "arxiv" not in clean.lower() and len(clean) > 5:
                    paper_title = clean
                    break
    except Exception as e:
        # Non-critical failure; we just proceed without a title
        debug_logs.append(f"Title extraction warning: {e}")

    try:
        for page_num in range(parser.page_count):
            page = parser.doc[page_num]
            # Fast skip for empty pages to save CPU cycles
            if not page.get_text().strip() and not page.get_drawings():
                continue

            # A. Extract Candidates (Spatial Clustering + Caption Linking)
            candidates = linker.extract_figures_from_page(arxiv_id, page_num)

            # --- Lazy BERT Inference Strategy ---
            # 1. Pre-filter using fast CPU rules (VisualFilter + Keyword Vetoes)
            surviving_candidates = []
            for cand in candidates:
                if classifier.is_hard_vetoed(cand):
                    continue
                surviving_candidates.append(cand)
            
            if not surviving_candidates:
                continue

            # B. Annotate with BERT (using the global worker-local model)
            # This is the heavy lifting step.
            if _worker_bert_helper:
                _worker_bert_helper.annotate(surviving_candidates)
            else:
                debug_logs.append(f"BERT model not initialized in worker {os.getpid()}")

            # C. Final Classification & Mining
            for cand in surviving_candidates:
                is_circuit = classifier.classify(cand)

                if not is_circuit:
                    continue

                # D. Mine Metadata (Passing the extracted Title!)
                try:
                    metadata = miner.mine(cand, paper_title=paper_title)
                except TypeError:
                    # Fallback compatibility if miner signature hasn't updated
                    metadata = miner.mine(cand)

                # E. Save Image IMMEDIATELY (Worker side I/O)
                # We save here to avoid passing heavy image bytes back to the main process
                safe_id = cand.arxiv_id.replace("/", "")
                filename = f"quantph_{safe_id}_p{cand.page_num}_f{cand.figure_idx}.png"
                output_path = Config.OUTPUT_IMAGE_DIR / filename
                
                with open(output_path, "wb") as f:
                    f.write(cand.image_bytes)
                
                # Only pass lightweight metadata back to Main
                mined_results.append((filename, metadata))
                valid_count += 1
                
    except Exception as e:
        debug_logs.append(f"CRITICAL ERROR in worker for {arxiv_id}: {str(e)}")
        # Return what we found so far, don't crash the pool
        return arxiv_id, 0, [], debug_logs

    return arxiv_id, valid_count, mined_results, debug_logs


class ImageStorageManager:
    """
    Manages the persistence of dataset metadata.
    
    This class is responsible for aggregating metadata records from multiple
    workers and atomically writing them to the master JSON file.
    """
    
    def __init__(self) -> None:
        self.metadata_store: Dict[str, Dict] = {}

    def register_metadata(self, filename: str, mined_metadata: Dict) -> None:
        """
        Records metadata for a file that has already been saved to disk.

        Parameters
        ----------
        filename : str
            The name of the image file (key).
        mined_metadata : Dict
            The dictionary of semantic data extracted from the figure.
        """
        self.metadata_store[filename] = mined_metadata

    def flush_json(self) -> None:
        """
        Writes the global metadata dictionary to the JSON output file.
        
        Uses an atomic write strategy (write to temp -> rename) to prevent
        file corruption if the process is interrupted.
        """
        temp_path = Config.OUTPUT_JSON_PATH.with_suffix('.tmp')
        with open(temp_path, "w") as f:
            json.dump(self.metadata_store, f, indent=4)
        os.replace(temp_path, Config.OUTPUT_JSON_PATH)
        print(f"  -> Metadata flushed to {Config.OUTPUT_JSON_PATH} ({len(self.metadata_store)} items)")


class PipelineOrchestrator:
    """
    The central controller for the extraction pipeline.

    Manages the lifecycle of the multiprocessing pool, feeds papers to
    workers, handles flow control (pausing downloads if workers are busy),
    and aggregates results.

    Attributes
    ----------
    list_manager : PaperListManager
        Handles the queue of arXiv IDs and tracking CSV.
    downloader : ArxivDownloader
        Handles downloading PDFs.
    storage : ImageStorageManager
        Handles saving JSON metadata.
    num_workers : int
        Number of concurrent worker processes (defaults to CPU count).
    """

    def __init__(self) -> None:
        self.list_manager = PaperListManager()
        self.downloader = ArxivDownloader()
        self.storage = ImageStorageManager()
        
        # Default to CPU count.
        self.num_workers = os.cpu_count() or 4
        
        # Define Cache Dir
        self.pdf_cache_dir = Config.BASE_DIR / "raw_pdfs"
        self.pdf_cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """
        Execute the main pipeline loop.

        This method keeps the ProcessPoolExecutor busy until the target
        number of images is collected or the paper list is exhausted.
        It balances downloading (I/O) with processing (CPU).
        """
        print(f"=== Starting Optimized Pipeline for Exam ID {Config.EXAM_ID} ===")
        print(f"Target: {Config.MAX_TOTAL_IMAGES} images.")
        print(f"Parallelism: Using {self.num_workers} worker processes.")

        total_collected = self.list_manager.get_total_images_collected()
        
        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=worker_init) as executor:
            futures = {} # Maps future -> arxiv_id
            
            while total_collected < Config.MAX_TOTAL_IMAGES:
                # 1. Flow Control: Check if we need to drain futures
                # If queue is full (2x workers), wait for space.
                # If queue is empty but list exhausted, wait for finish.
                can_queue = len(futures) < self.num_workers * 2
                
                arxiv_id = None
                if can_queue:
                    # Try to get next paper. 
                    # NOTE: get_next_paper() returns None if paper is locked (in_progress)
                    # or if the list is finished.
                    arxiv_id = self.list_manager.get_next_paper()
                
                # 2. Logic to Decide: Download, Wait, or Stop
                if arxiv_id:
                    # Case A: We have a new paper and capacity to process it
                    pdf_path = self._get_or_download_pdf(arxiv_id)
                    
                    if not pdf_path:
                        print(f"  Skipping {arxiv_id} (Download failed)")
                        self.list_manager.mark_as_failed(arxiv_id) # Releases lock
                        continue

                    print(f"  [Queueing] {arxiv_id}")
                    future = executor.submit(process_paper_task, arxiv_id, pdf_path)
                    futures[future] = arxiv_id
                
                else:
                    # Case B: No paper returned. 
                    # Either list is empty OR all remaining papers are currently processing.
                    if not futures:
                        print("No more papers and no active jobs. Pipeline stopping.")
                        break
                    
                    # We must wait for at least one job to finish to free up a slot 
                    # or finish the workload.
                    # wait(futures, return_when=FIRST_COMPLETED) avoids spin-looping CPU.
                    done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                    
                    # We continue the loop; processing happens in step 3 below.

                # 3. Process Completed Jobs
                self._process_completed_futures(futures)
                
                # Update total collected for loop condition
                total_collected = self.list_manager.get_total_images_collected()

            # End of while loop: Drain remaining futures
            print("\nTarget reached or list exhausted. Waiting for remaining jobs...")
            self._process_completed_futures(futures, wait_for_all=True)

        print("\n=== Pipeline Finished ===")

    def _get_or_download_pdf(self, arxiv_id: str) -> Optional[Path]:
        """
        Check local cache for PDF; if missing, download it.

        Parameters
        ----------
        arxiv_id : str
            The arXiv ID to fetch.

        Returns
        -------
        Optional[Path]
            Path to the PDF if successful, None otherwise.
        """
        cached_path = self.pdf_cache_dir / f"{arxiv_id}.pdf"
        if cached_path.exists() and cached_path.stat().st_size > 0:
            return cached_path
        
        return self.downloader.get_pdf_path(arxiv_id)

    def _process_completed_futures(self, futures: dict, wait_for_all: bool = False) -> None:
        """
        Collect results from completed futures and update state.

        This method removes finished jobs from the 'futures' dict,
        updates the PaperListManager (releasing locks), and merges
        metadata into the storage.

        Parameters
        ----------
        futures : dict
            The dictionary mapping {Future: arxiv_id}. Modified in-place.
        wait_for_all : bool
            If True, blocks until ALL futures are complete.
            If False, only processes futures that are currently done.
        """
        if not futures:
            return

        # Identify which futures are done
        futures_to_check = []
        if wait_for_all:
             # as_completed yields futures as they finish (blocking)
             for f in as_completed(futures.keys()):
                 futures_to_check.append(f)
        else:
            # Non-blocking scan of current list
            for f in list(futures.keys()):
                if f.done():
                    futures_to_check.append(f)
        
        for future in futures_to_check:
            arxiv_id = futures.pop(future) # Remove from active list
            try:
                # Retrieve result from worker
                _, count, metadata_list, logs = future.result()
                
                for log in logs:
                    print(f"    [Worker Log] {log}")

                # Update Main State (This RELEASES the in_progress lock in manager)
                self.list_manager.update_count(arxiv_id, count)
                
                # Merge Metadata
                for filename, meta in metadata_list:
                    self.storage.register_metadata(filename, meta)
                
                if count > 0:
                    print(f"  -> Finished {arxiv_id}: {count} circuits.")
                    self.storage.flush_json()
                else:
                    print(f"  -> Finished {arxiv_id}: No circuits found.")
                    
            except Exception as e:
                print(f"  CRITICAL MAIN ERROR processing {arxiv_id}: {e}")
                self.list_manager.mark_as_failed(arxiv_id)


if __name__ == "__main__":
    Config.setup_directories()
    start_time = time.time()
    
    orchestrator = PipelineOrchestrator()
    orchestrator.run()
    
    duration = time.time() - start_time
    print(f"Total Runtime: {duration / 60:.2f} minutes")