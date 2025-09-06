import chromadb
import openai
import streamlit as st
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from config import (
    COLLECTION_NAME,
    SIMILARITY_SEARCH_RESULTS,
    EMBEDDING_MODEL,
    DISTANCE_FUNCTION,
)


class ChromaVectorDB:
    def __init__(self):
        self.openai_client = None
        self.chroma_client = None
        self.collection = None

    def initialize(self, openai_api_key: str) -> bool:
        try:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            self.openai_client.models.list()

            # Initialize ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(path="chromadb_collections")

            try:
                self.collection = self.chroma_client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": DISTANCE_FUNCTION},
                )
                print("ChromaDB collection created successfully =============>")
            except Exception:
                self.collection = self.chroma_client.get_collection(
                    name=COLLECTION_NAME
                )
                print("Connected to existing ChromaDB collection ===============>")

            return True

        except Exception as e:
            print(f"Error initializing vector database =======> {str(e)}")
            return False

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        try:
            if not text.strip():
                return None

            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL, input=text, encoding_format="float"
            )
            # print(
            #     "Generated embedding successfully =============>",
            #     response.data[0].embedding,
            # )
            return response.data[0].embedding

        except Exception as e:
            st.error(f"Error generating embedding: {str(e)}")
            return None

    def add_documents(
        self, text_chunks: List[str], source_type: str, source_name: str
    ) -> bool:
        try:
            if not text_chunks or not self.collection:
                return False

            processed_chunks = 0
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, chunk in enumerate(text_chunks):
                if not chunk.strip():
                    continue

                # Chunking progress Bar
                progress = (i + 1) / len(text_chunks)
                progress_bar.progress(progress)
                status_text.text(
                    f"Embedding chunk {i + 1} of {len(text_chunks)} ========>"
                )

                embedding = self.generate_embedding(chunk)
                if embedding is None:
                    continue

                doc_id = f"{source_type}_{source_name}_{i}_{uuid.uuid4().hex[:8]}"

                self.collection.add(
                    documents=[chunk],  # Original text
                    embeddings=[embedding],  # Vector representation
                    metadatas=[
                        {  # Metadata for filtering and tracking
                            "source_type": source_type,
                            "source_name": source_name,
                            "chunk_index": i,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ],
                    ids=[doc_id],  # Unique identifier
                )
                processed_chunks += 1

            progress_bar.empty()
            status_text.empty()

            if processed_chunks > 0:
                st.success(
                    f"==========> Successfully stored {processed_chunks} chunks in ChromaDB"
                )

                # with st.expander("Vector Database Details"):
                #     st.info(f"""
                #     **Vector Storage Complete:**
                #     - Chunks processed: {processed_chunks}
                #     - Embedding dimensions: 1536 (OpenAI text-embedding-3-small)
                #     - Distance function: {DISTANCE_FUNCTION}
                #     - Collection: {COLLECTION_NAME}
                #     """)
                return True
            else:
                st.warning("No valid chunks were processed")
                return False

        except Exception as e:
            st.error(f"Error adding documents to vector database: {str(e)}")
            return False

    def similarity_search(
        self, query: str, n_results: int = SIMILARITY_SEARCH_RESULTS
    ) -> List[Dict[str, Any]]:
        try:
            if not self.collection:
                st.warning("Vector database not initialized")
                return []

            # Generate embedding for search query
            query_embedding = self.generate_embedding(query)
            if query_embedding is None:
                return []

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            search_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    similarity_score = (
                        1 - results["distances"][0][i] if results["distances"] else None
                    )

                    search_results.append(
                        {
                            "document": doc,
                            "metadata": results["metadatas"][0][i]
                            if results["metadatas"]
                            else {},
                            "similarity_score": similarity_score,
                            "distance": results["distances"][0][i]
                            if results["distances"]
                            else None,
                        }
                    )

            return search_results

        except Exception as e:
            st.error(f"Error performing similarity search: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection"""
        try:
            if not self.collection:
                return {"total_documents": 0, "collection_exists": False}

            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_exists": True,
                "collection_name": COLLECTION_NAME,
                "distance_function": DISTANCE_FUNCTION,
            }

        except Exception as e:
            return {"total_documents": 0, "collection_exists": False}

    def delete_all_documents(self) -> bool:
        """Clear all documents from ChromaDB collection"""
        try:
            if not self.collection:
                return False
            all_docs = self.collection.get()

            if all_docs["ids"]:
                self.collection.delete(ids=all_docs["ids"])
                st.success(f"Deleted {len(all_docs['ids'])} documents from ChromaDB")
            else:
                st.info("No documents to delete")

            return True

        except Exception as e:
            st.error(f"Error clearing vector database: {str(e)}")
            return False

    def get_documents_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source"""
        try:
            if not self.collection:
                return []

            results = self.collection.get(
                where={"source_name": source_name}, include=["documents", "metadatas"]
            )

            documents = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    documents.append(
                        {
                            "document": doc,
                            "metadata": results["metadatas"][i]
                            if results["metadatas"]
                            else {},
                        }
                    )

            return documents

        except Exception as e:
            st.error(f"Error retrieving documents by source: {str(e)}")
            return []
