"""
data_management.py - I/O, State Tracking, and Robust Downloading.

This module handles the 'Information Retrieval' and 'Project Management'
aspects of the pipeline. It ensures the process is resumable,
fault-tolerant, and thread-safe when managing job queues.
"""

import csv
import time
import requests
from typing import List, Dict, Optional, Any
from pathlib import Path
from config import Config


class PaperListManager:
    """
    Manages the queue of arXiv papers and tracks processing progress.

    This class abstracts the CSV reading/writing logic, ensuring that
    the pipeline can be stopped and resumed without losing data.
    It also implements a locking mechanism to prevent multiple workers
    from processing the same paper simultaneously.

    Attributes
    ----------
    paper_status : Dict[str, Optional[int]]
        A dictionary mapping arXiv IDs to the number of images found.
        Value is None if the paper has not been processed yet.
    queue : List[str]
        The ordered list of arXiv IDs to process.
    in_progress : Set[str]
        A set tracking papers currently assigned to workers to prevent
        duplicate queuing in a multiprocessing environment.
    """

    def __init__(self) -> None:
        """
        Initialize the manager by loading the paper list and existing progress.

        Raises
        ------
        FileNotFoundError
            If the input paper list file defined in Config.PAPER_LIST_PATH
            does not exist.
        """
        self.paper_status: Dict[str, Optional[int]] = {}
        self.queue: List[str] = []
        self.in_progress = set()  # Tracks active jobs
        self._load_data()

    def _load_data(self) -> None:
        """
        Load the target paper list and sync with the existing CSV tracking file.

        This method:
        1. Reads the raw text file of arXiv IDs.
        2. Sanitizes the IDs (removes 'arXiv:' prefix).
        3. Loads previous results from the output CSV to resume progress.
        """
        if not Config.PAPER_LIST_PATH.exists():
            raise FileNotFoundError(f"Missing paper list: {Config.PAPER_LIST_PATH}")
        
        # 1. Load the Queue from the text file
        with open(Config.PAPER_LIST_PATH, 'r') as f:
            self.queue = []
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                
                # Sanitize the ID (e.g., "arXiv:1234.5678" -> "1234.5678")
                clean_id = raw.lower().replace("arxiv:", "").strip()
                self.queue.append(clean_id)

        # Initialize status for all papers as None (unprocessed)
        for pid in self.queue:
            self.paper_status[pid] = None

        # 2. Sync with existing CSV progress (if resuming a run)
        if Config.OUTPUT_CSV_PATH.exists():
            with open(Config.OUTPUT_CSV_PATH, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip the header row
                for row in reader:
                    if len(row) < 2: 
                        continue
                    pid, count_str = row[0], row[1]
                    
                    # Update status if this ID is in our current queue
                    if pid in self.paper_status:
                        if count_str.strip() == "":
                            self.paper_status[pid] = None
                        else:
                            self.paper_status[pid] = int(count_str)

    def get_next_paper(self) -> Optional[str]:
        """
        Retrieve the next unprocessed AND unassigned arXiv ID from the queue.

        This method implements a 'locking' strategy. When a paper is returned,
        it is added to `self.in_progress` so it is not returned again
        until it is either completed or marked as failed.

        Returns
        -------
        Optional[str]
            The arXiv ID string (e.g., '2101.00001') if available.
            None if all papers are finished or currently in progress.
        """
        for arxiv_id in self.queue:
            # Check if paper is:
            # 1. Unfinished (status is None)
            # 2. Not currently running in a worker (not in in_progress)
            if self.paper_status[arxiv_id] is None and arxiv_id not in self.in_progress:
                self.in_progress.add(arxiv_id)  # Lock the paper
                return arxiv_id
        return None

    def update_count(self, arxiv_id: str, count: int) -> None:
        """
        Update the image count for a specific paper and save to disk.

        This method also releases the 'in_progress' lock for the paper.

        Parameters
        ----------
        arxiv_id : str
            The arXiv ID of the processed paper.
        count : int
            Number of valid images extracted.
        """
        self.paper_status[arxiv_id] = count
        if arxiv_id in self.in_progress:
            self.in_progress.remove(arxiv_id)  # Unlock the paper
        self._save_csv()

    def mark_as_failed(self, arxiv_id: str) -> None:
        """
        Mark a paper as processed with 0 images due to a fatal error.

        This ensures the pipeline doesn't get stuck retrying a broken file.

        Parameters
        ----------
        arxiv_id : str
            The arXiv ID that failed.
        """
        # Record 0 to satisfy the CSV requirement (integers only).
        self.paper_status[arxiv_id] = 0
        if arxiv_id in self.in_progress:
            self.in_progress.remove(arxiv_id)  # Unlock the paper
        self._save_csv()

    def _save_csv(self) -> None:
        """
        Write the current status of all papers to the CSV file.

        The CSV is rewritten entirely to ensure consistency. It follows
        the order of the original input queue.
        """
        with open(Config.OUTPUT_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["arxiv_id", "count"])
            
            for arxiv_id in self.queue:
                status = self.paper_status.get(arxiv_id)
                
                # Write empty string if None (unprocessed), else write number
                val = "" if status is None else str(status)
                writer.writerow([arxiv_id, val])
    
    def get_total_images_collected(self) -> int:
        """
        Calculate total images collected across all processed papers.

        Returns
        -------
        int
            Sum of image counts for all finished papers.
        """
        return sum(c for c in self.paper_status.values() if c is not None)


class ArxivDownloader:
    """
    Handles secure and robust downloading of PDFs from arXiv.

    Implements caching and exponential backoff retry logic to handle
    network flakiness and avoid overloading the arXiv servers.

    Attributes
    ----------
    BASE_URL : str
        The template URL for fetching PDFs from arXiv.
    """
    
    BASE_URL = "https://arxiv.org/pdf/{}.pdf"

    def get_pdf_path(self, arxiv_id: str) -> Optional[Path]:
        """
        Get the local path for a PDF, downloading it if necessary.

        This method checks the local cache first. If the file exists and
        is valid (non-zero size), it returns the path. Otherwise, it
        attempts to download it.

        Parameters
        ----------
        arxiv_id : str
            The arXiv ID to retrieve.

        Returns
        -------
        Optional[Path]
            Path object to the local PDF file if successful.
            None if download failed after retries.
        """
        target_path = Config.PDF_CACHE_DIR / f"{arxiv_id}.pdf"
        
        # Check Cache
        if target_path.exists():
            # Validation: Check for 0-byte corrupted files
            if target_path.stat().st_size > 0:
                return target_path
            else:
                # Remove corrupted file and retry download
                target_path.unlink()
            
        return self._download_with_retry(arxiv_id, target_path)

    def _download_with_retry(self, arxiv_id: str, target_path: Path) -> Optional[Path]:
        """
        Internal method to download a file with exponential backoff.

        Parameters
        ----------
        arxiv_id : str
            ID for URL construction.
        target_path : Path
            Destination for the file.

        Returns
        -------
        Optional[Path]
            Path if successful, None otherwise.
        """
        url = self.BASE_URL.format(arxiv_id)
        
        for attempt in range(1, Config.MAX_RETRIES + 1):
            try:
                print(f"  Downloading {arxiv_id} (Attempt {attempt}/{Config.MAX_RETRIES})...")
                
                # User-Agent is polite practice for academic scraping
                headers = {'User-Agent': 'UniversityProject/1.0 (Educational Use)'}
                response = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
                
                if response.status_code == 200:
                    # Validate PDF Magic Bytes (starts with %PDF)
                    # This prevents saving HTML error pages (e.g., Cloudflare blocks) as .pdf
                    if not response.content.startswith(b"%PDF"):
                        print(f"  Error: Server returned non-PDF content for {arxiv_id}")
                        return None
                        
                    with open(target_path, "wb") as f:
                        f.write(response.content)
                    
                    # Politeness delay to respect server rate limits
                    time.sleep(Config.REQUEST_DELAY)
                    return target_path
                
                elif response.status_code == 404:
                    print(f"  Error 404: Paper {arxiv_id} not found.")
                    return None # No point retrying a 404
                
                else:
                    print(f"  Server returned status {response.status_code}")
            
            except requests.RequestException as e:
                print(f"  Network error: {e}")

            # Backoff Logic: Wait before retrying (unless it's the last attempt)
            if attempt < Config.MAX_RETRIES:
                sleep_time = Config.REQUEST_DELAY * (Config.BACKOFF_FACTOR ** (attempt - 1))
                print(f"  Retrying in {sleep_time}s...")
                time.sleep(sleep_time)

        print(f"  Failed to download {arxiv_id} after {Config.MAX_RETRIES} attempts.")
        return None