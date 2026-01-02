import os
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from reasoning.tools.pinecone_bridge import get_pinecone_engine
from mcp_server.registry import ToolRegistry

class PDFIngestor:
    """
    Handles deep ingestion of PDF documents into Pinecone vector memory.
    """
    def __init__(self):
        self.pinecone = get_pinecone_engine()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

    @ToolRegistry.register(name="ingest_pdf")
    def ingest(self, file_path: str) -> str:
        """
        Parses a PDF file from the local path, splits it into semantic chunks,
        and indexes it into Pinecone.
        """
        if not os.path.exists(file_path):
            return f"ERROR: File not found at {file_path}"
        
        if not file_path.lower().endswith(".pdf"):
            return "ERROR: Only PDF files are supported for deep ingestion."

        print(f"PDF_INGESTOR: Processing {file_path}...")
        
        try:
            # 1. Extract Text
            doc = fitz.open(file_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()

            if not full_text.strip():
                return "ERROR: No text extracted from PDF. Possibly an image-only skip."

            # 2. Split into Chunks
            chunks = self.text_splitter.split_text(full_text)
            print(f"PDF_INGESTOR: Split into {len(chunks)} chunks.")

            # 3. Ingest into Pinecone
            if not self.pinecone.online:
                return "ERROR: Pinecone is offline. Cannot perform deep ingestion."

            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": os.path.basename(file_path),
                    "chunk_idx": i,
                    "type": "scientific_paper"
                }
                self.pinecone.upsert(chunk, metadata=metadata)

            return f"SUCCESS: Ingested '{os.path.basename(file_path)}' ({len(chunks)} chunks) into Deep Memory."

        except Exception as e:
            return f"SYSTEM ERROR during PDF ingestion: {str(e)}"

# Registering the tool automatically via the class import in orchestrator
_ingestor = PDFIngestor()
