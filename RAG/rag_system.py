import pandas as pd
from google import genai
from google.genai import types
import streamlit as st
from typing import List, Dict, Any, Optional
from vector_database import ChromaVectorDB
from web_search import WebSearcher
from document_processor import DocumentProcessor
from config import GEMINI_MODEL, MAX_TOKENS, TEMPERATURE, GEMINI_API_KEY, SERPER_API_KEY


class RAGSystem:
    def __init__(self):
        self.vector_db = ChromaVectorDB()
        self.web_searcher = None
        self.doc_processor = DocumentProcessor()
        self.client = None

    def initialize(
        self, gemini_api_key: str, serper_api_key: Optional[str] = None
    ) -> bool:
        try:
            # Initialize vector database with ChromaDB
            if not self.vector_db.initialize(gemini_api_key):
                print("=======> Failed to initialize ChromaDB vector database")
                return False

            self.client = genai.Client(api_key=gemini_api_key)
            
            print("Gemini connection successful =============>")

            if serper_api_key:
                self.web_searcher = WebSearcher(serper_api_key)
                print("RAG system initialized with web search ")
            else:
                self.web_searcher = WebSearcher()  # Disabled web searcher
                print("RAG system initialized (no web search)")

            return True

        except Exception as e:
            print(f"Error initializing RAG system=============> {str(e)}")
            return False

    def add_document(self, text: str, source_type: str, source_name: str) -> bool:
        if not text or not text.strip():
            st.warning("Empty document provided")
            return False

        # Chunk the document text
        chunks = self.doc_processor.chunk_text(text)

        if not chunks:
            st.warning("No valid chunks created from document")
            return False

        # Add chunks to vector database
        return self.vector_db.add_documents(chunks, source_type, source_name)

    def retrieve_context(
        self, query: str, include_web_search: bool = False
    ) -> Dict[str, Any]:
        context_parts = []
        sources = []

        # Vector database retrieval using ChromaDB similarity search
        vector_results = self.vector_db.similarity_search(query)

        if vector_results:
            print("Found relevant chunks in vector database")

            for result in vector_results:
                context_parts.append(result["document"])
                sources.append(
                    {
                        "type": "vector_db",
                        "source": result["metadata"].get("source_name", "Unknown"),
                        "similarity": result.get("similarity_score", 0),
                    }
                )

        # Web search retrieval (if enabled and requested)
        web_results = []
        if include_web_search and self.web_searcher and self.web_searcher.enabled:
            web_results = self.web_searcher.search(query)

            if web_results:
                st.info(f"Found {len(web_results)} web search results")

                # Add web results to context
                for result in web_results:
                    context_parts.append(
                        f"Web Result: {result['title']} - {result['snippet']}"
                    )
                    sources.append(
                        {
                            "type": "web_search",
                            "source": result["title"],
                            "link": result["link"],
                        }
                    )

        return {
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "vector_results": len(vector_results),
            "web_results": len(web_results),
        }

    def generate_response(
        self, query: str, include_web_search: bool = False
    ) -> Dict[str, Any]:
        try:
            retrieval_results = self.retrieve_context(query, include_web_search)

            system_prompt = self._build_system_prompt(
                retrieval_results["context"], query
            )

            # Define tool configuration if needed (currently none in standard response)
            # You can pass tools=[] to generate_content if required.
            
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=system_prompt, # Using system prompt as content for now or pass as system_instruction
                config=types.GenerateContentConfig(
                    max_output_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                ),
            )

            generated_text = response.text

            return {
                "response": generated_text,
                "context_used": retrieval_results["context"],
                "sources": retrieval_results["sources"],
                "vector_results_count": retrieval_results["vector_results"],
                "web_results_count": retrieval_results["web_results"],
            }

        except Exception as e:
            st.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error generating a response. Please try again.",
                "context_used": "",
                "sources": [],
                "vector_results_count": 0,
                "web_results_count": 0,
            }

    def _build_system_prompt(self, context: str, query: str) -> str:
        base_prompt = f"""You are a helpful AI assistant with access to a knowledge base and web search.

CONTEXT FROM KNOWLEDGE BASE:
{context}

INSTRUCTIONS:
1. Use the provided context to answer the user's question accurately
2. If the context contains relevant information, prioritize it in your response
3. If the context doesn't fully answer the question, provide general knowledge while noting limitations
4. Be clear about what information comes from the knowledge base vs general knowledge
5. Provide specific and helpful answers

USER QUESTION: {query}

Please provide a comprehensive response based on the available context."""

        return base_prompt

    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall RAG system statistics"""
        vector_stats = self.vector_db.get_collection_stats()

        return {
            "vector_db_documents": vector_stats.get("total_documents", 0),
            "vector_db_active": vector_stats.get("collection_exists", False),
            "web_search_enabled": self.web_searcher.enabled
            if self.web_searcher
            else False,
            "collection_name": vector_stats.get("collection_name", ""),
            "distance_function": vector_stats.get("distance_function", ""),
        }

    def clear_knowledge_base(self) -> bool:
        """Clear all documents from the vector database"""
        return self.vector_db.delete_all_documents()
