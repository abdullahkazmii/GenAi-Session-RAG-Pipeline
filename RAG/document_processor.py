import PyPDF2
import requests
from bs4 import BeautifulSoup
import streamlit as st
from typing import List, Optional, Tuple
from config import CHUNK_SIZE, CHUNK_OVERLAP, WEB_HEADERS


class DocumentProcessor:
    @staticmethod
    def chunk_text(
        text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
    ) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval

        Why chunking is important for RAG:
        - Large documents can't fit in LLM context windows
        - Smaller chunks improve retrieval precision
        - Overlapping chunks ensure context isn't lost at boundaries
        """
        if not text or not text.strip():
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk.strip())

            start = end - overlap

            if start >= len(text):
                break

        return chunks

    @staticmethod
    def extract_pdf_text(pdf_file) -> Optional[str]:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            total_pages = len(pdf_reader.pages)

            if total_pages == 0:
                st.warning("PDF file appears to be empty")
                return None

            progress_bar = st.progress(0)

            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                progress_bar.progress((i + 1) / total_pages)

            progress_bar.empty()

            if not text.strip():
                st.warning("No text could be extracted from PDF")
                return None

            return text.strip()

        except Exception as e:
            st.error(f"Error extracting PDF text: {str(e)}")
            return None

    @staticmethod
    def extract_website_content(url: str) -> Optional[Tuple[str, str]]:
        try:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            response = requests.get(url, headers=WEB_HEADERS, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove unwanted elements
            for element in soup(
                ["script", "style", "nav", "footer", "header", "aside"]
            ):
                element.decompose()

            title = soup.title.string.strip() if soup.title else "Website Content"

            main_content = (
                soup.find("main") or soup.find("article") or soup.find("body")
            )

            if main_content:
                text = main_content.get_text(separator=" ", strip=True)
            else:
                text = soup.get_text(separator=" ", strip=True)

            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            cleaned_text = " ".join(chunk for chunk in chunks if chunk)

            if not cleaned_text.strip():
                st.warning("No content could be extracted from website")
                return None

            return cleaned_text, title

        except Exception as e:
            st.error(f"Error extracting website content: {str(e)}")
            return None
