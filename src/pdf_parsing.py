"""
pdf_parsing.py - Low-level PDF page parsing with PyMuPDF.

This module abstracts the technical details of the PyMuPDF (fitz) library.
It provides a unified interface to extract:
1.  **Text Blocks**: Positioned text for caption analysis.
2.  **Vector Graphics**: Using a clustering algorithm to group separate lines
    into coherent figure regions.
3.  **Raster Images**: Embedded bitmaps (PNG/JPG).
4.  **Page Rendering**: Converting specific page regions into PNG bytes.
"""

import pymupdf as fitz 
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path


@dataclass
class TextBlock:
    """
    Represents a discrete block of text on a PDF page.

    Attributes
    ----------
    block_id : int
        Unique index of the block on the page.
    bbox : Tuple[float, float, float, float]
        The bounding box coordinates (x0, y0, x1, y1).
    text : str
        The textual content of the block.
    """
    block_id: int
    bbox: Tuple[float, float, float, float]
    text: str


@dataclass
class ImageInfo:
    """
    Represents a raster image found on a PDF page.

    Attributes
    ----------
    bbox : Tuple[float, float, float, float]
        The bounding box coordinates (x0, y0, x1, y1).
    image_bytes : bytes
        The raw byte content of the extracted image.
    """
    bbox: Tuple[float, float, float, float]
    image_bytes: bytes


class PDFPageParser:
    """
    A wrapper around PyMuPDF to parse and render PDF content.

    This class handles the low-level extraction of visual and textual
    elements, normalizing coordinates and formats for the upstream pipeline.

    Attributes
    ----------
    doc : fitz.Document
        The open PyMuPDF document object.
    page_count : int
        Total number of pages in the document.
    """

    def __init__(self, doc_path: Path) -> None:
        """
        Initialize the parser by opening the PDF file.

        Parameters
        ----------
        doc_path : Path
            Filesystem path to the PDF file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not doc_path.exists():
            raise FileNotFoundError(f"PDF not found: {doc_path}")
        self.doc = fitz.open(doc_path)
        self.page_count = len(self.doc)

    def parse_page(self, page_num: int) -> Dict:
        """
        Parse a single page to extract all relevant content.

        This method performs three distinct extraction passes:
        1.  **Text Extraction**: Gets text blocks for caption linking.
        2.  **Vector Clustering**: logical grouping of vector paths (IAAA).
        3.  **Image Extraction**: Finding embedded raster images.

        Parameters
        ----------
        page_num : int
            The 0-based page index.

        Returns
        -------
        Dict
            A dictionary containing:
            - 'page_width', 'page_height': Dimensions in points.
            - 'text_blocks': List[TextBlock].
            - 'page_text': Full raw text of the page.
            - 'vector_clusters': List[Tuple[x0, y0, x1, y1]].
            - 'images': List[ImageInfo].
        """
        page = self.doc[page_num]
        
        # 1. Text extraction
        # "blocks" mode groups text into paragraphs/columns automatically
        raw_blocks = page.get_text("blocks", sort=True)
        text_blocks = []
        page_text_parts = []
        
        for i, b in enumerate(raw_blocks):
            # b structure: (x0, y0, x1, y1, text, block_no, block_type)
            clean_text = b[4].strip()
            text_blocks.append(TextBlock(i, (b[0], b[1], b[2], b[3]), clean_text))
            if clean_text:
                page_text_parts.append(clean_text)

        full_page_text = "\n\n".join(page_text_parts)

        # 2. Vector Clustering (IAAA Algorithm)
        # Groups separate vector lines into cohesive figure regions.
        vector_clusters = self._cluster_vector_paths(page)
        
        # 3. Raster Image Extraction
        embedded_images = self._extract_images(page)

        return {
            'page_width': page.rect.width,
            'page_height': page.rect.height,
            'text_blocks': text_blocks,
            'page_text': full_page_text,
            'vector_clusters': vector_clusters, 
            'images': embedded_images
        }

    def _cluster_vector_paths(self, page: fitz.Page, margin: float = 15.0) -> List[Tuple[float, float, float, float]]:
        """
        Group vector drawing paths into clusters using IAAA.
        
        **IAAA (Iterative Adjacency Area Aggregation)**:
        Quantum circuits in PDFs are often drawn as hundreds of individual
        lines and rectangles. This algorithm:
        1.  Starts with a bounding box for every single line.
        2.  Expands each box by a `margin`.
        3.  Iteratively merges any boxes that overlap.
        4.  Repeats until no more boxes overlap.

        Parameters
        ----------
        page : fitz.Page
            The page object to scan.
        margin : float, optional
            The proximity threshold (in points) to consider two paths as 
            part of the same figure. Default is 15.0.

        Returns
        -------
        List[Tuple[float, float, float, float]]
            A list of bounding boxes (x0, y0, x1, y1) representing 
            clustered vector regions.
        """
        paths = page.get_drawings()
        
        # Optimization: Skip clustering on extremely complex pages (e.g., full-page scatter plots)
        # to prevent O(N^2) performance degradation.
        if len(paths) > 3000: 
            return []
            
        # Initialize one box per drawing path
        boxes = [fitz.Rect(p["rect"]) for p in paths if p["rect"].width > 0 or p["rect"].height > 0]
        if not boxes: return []

        # The IAAA Loop
        changed = True
        while changed:
            changed = False
            new_boxes = []
            while boxes:
                # Pop the first box as the 'seed'
                current = boxes.pop(0)
                
                # Create a search area expanded by the margin
                search_area = fitz.Rect(current.x0 - margin, current.y0 - margin, 
                                        current.x1 + margin, current.y1 + margin)
                
                # Check against all remaining boxes
                i = 0
                while i < len(boxes):
                    other = boxes[i]
                    if search_area.intersects(other):
                        # Merge them!
                        current.include_rect(other)
                        # Update search area for the grown box
                        search_area = fitz.Rect(current.x0 - margin, current.y0 - margin, 
                                                current.x1 + margin, current.y1 + margin)
                        # Remove the merged box from the pool
                        boxes.pop(i)
                        changed = True
                    else:
                        i += 1
                new_boxes.append(current)
            boxes = new_boxes

        return [(b.x0, b.y0, b.x1, b.y1) for b in boxes]

    def _extract_images(self, page: fitz.Page) -> List[ImageInfo]:
        """
        Identify and extract embedded raster images.

        This handles standard PDF images (XObjects).

        Parameters
        ----------
        page : fitz.Page
            The page to scan.

        Returns
        -------
        List[ImageInfo]
            A list of image objects containing bounding boxes and raw bytes.
        """
        images = []
        image_list = page.get_images(full=True)
        
        for img in image_list:
            xref = img[0] # The internal reference ID
            try:
                # Get the location of the image on the page
                bbox_rect = page.get_image_bbox(xref)
                if bbox_rect.is_empty: 
                    continue
                
                # Extract raw bytes
                base_image = self.doc.extract_image(xref)
                images.append(ImageInfo(
                    (bbox_rect.x0, bbox_rect.y0, bbox_rect.x1, bbox_rect.y1), 
                    base_image["image"]
                ))
            except Exception:
                # Gracefully skip corrupted images
                continue
        return images

    def render_page_clip(self, page_num: int, bbox: Tuple[float, float, float, float], zoom: float = 2.0) -> Optional[bytes]:
        """
        Render a specific region of a page to a PNG image.

        Used to generate the final output images from the merged bounding boxes.

        Parameters
        ----------
        page_num : int
            The 0-based page index.
        bbox : Tuple[float, float, float, float]
            The region to render (x0, y0, x1, y1).
        zoom : float, optional
            The scaling factor. 2.0 = 144 DPI (Double standard 72 DPI).
            Default is 2.0.

        Returns
        -------
        Optional[bytes]
            The PNG image data as bytes, or None if rendering fails.
        """
        if page_num >= self.page_count: return None
        try:
            page = self.doc[page_num]
            mat = fitz.Matrix(zoom, zoom)
            
            # alpha=False ensures a white background instead of transparent
            pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(bbox), alpha=False)
            return pix.tobytes("png")
        except Exception as e:
            return None