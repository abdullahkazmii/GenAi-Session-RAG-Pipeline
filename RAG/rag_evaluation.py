"""
RAG Evaluation using RAGAS (RAG Assessment)

RAGAS provides metrics to evaluate RAG system performance:
- Faithfulness: How well the answer is grounded in the retrieved context
- Answer Relevancy: How relevant the answer is to the question
- Context Precision: How precise the retrieved context is
- Context Recall: How complete the retrieved context is

This module provides evaluation capabilities for RAG system quality assessment.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class RAGEvaluator:
    """
    RAG Evaluation using RAGAS framework

    RAGAS (RAG Assessment) evaluates:
    1. Faithfulness: Is the answer based on retrieved context?
    2. Answer Relevancy: Does the answer address the question?
    3. Context Precision: Is retrieved context relevant?
    4. Context Recall: Is all relevant context retrieved?
    """

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.evaluation_history = []

    def evaluate_faithfulness(self, question: str, answer: str, context: str) -> float:
        """
        Evaluate faithfulness: How well is the answer grounded in context?

        Simple faithfulness check:
        - Does answer contain information from context?
        - Are there unsupported claims in the answer?

        Note: Full RAGAS implementation requires additional models for evaluation
        """
        if not context or not answer:
            return 0.0

        # Simple keyword overlap check (simplified version)
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())

        # Remove common stop words for better evaluation
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        context_words = context_words - stop_words
        answer_words = answer_words - stop_words

        if not answer_words:
            return 0.0

        # Calculate overlap ratio
        overlap = len(context_words.intersection(answer_words))
        faithfulness_score = min(overlap / len(answer_words), 1.0)

        return faithfulness_score

    def evaluate_relevancy(self, question: str, answer: str) -> float:
        """
        Evaluate answer relevancy: How well does answer address the question?

        Simple relevancy check:
        - Keyword overlap between question and answer
        - Answer length appropriateness
        """
        if not question or not answer:
            return 0.0

        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        # Remove stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "what",
            "how",
            "when",
            "where",
            "why",
        }
        question_words = question_words - stop_words
        answer_words = answer_words - stop_words

        if not question_words:
            return 0.5  # Neutral score if no meaningful question words

        # Calculate relevancy based on word overlap
        overlap = len(question_words.intersection(answer_words))
        relevancy_score = min(overlap / len(question_words), 1.0)

        # Bonus for appropriate answer length
        if 50 <= len(answer) <= 500:  # Reasonable answer length
            relevancy_score = min(relevancy_score + 0.1, 1.0)

        return relevancy_score

    def evaluate_context_precision(
        self, question: str, retrieved_chunks: List[str]
    ) -> float:
        """
        Evaluate context precision: How relevant is the retrieved context?

        Measures:
        - Proportion of retrieved chunks that are relevant to question
        - Quality of vector database retrieval
        """
        if not retrieved_chunks or not question:
            return 0.0

        question_words = set(question.lower().split())
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        question_words = question_words - stop_words

        relevant_chunks = 0

        for chunk in retrieved_chunks:
            chunk_words = set(chunk.lower().split()) - stop_words
            overlap = len(question_words.intersection(chunk_words))

            # Consider chunk relevant if it has significant overlap with question
            if overlap >= 2:  # At least 2 meaningful word matches
                relevant_chunks += 1

        precision = relevant_chunks / len(retrieved_chunks)
        return precision

    def run_evaluation(
        self, test_questions: List[str], include_web_search: bool = False
    ) -> pd.DataFrame:
        """
        Run complete RAG evaluation on test questions

        Returns evaluation metrics:
        - Faithfulness scores
        - Relevancy scores
        - Context precision scores
        - Response times
        """
        evaluation_results = []

        progress_bar = st.progress(0)

        for i, question in enumerate(test_questions):
            with st.spinner(f"Evaluating question {i + 1}/{len(test_questions)}..."):
                start_time = datetime.now()

                # Generate response using RAG system
                rag_result = self.rag_system.generate_response(
                    question, include_web_search
                )

                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()

                # Retrieve context for evaluation
                retrieval_results = self.rag_system.retrieve_context(
                    question, include_web_search
                )
                retrieved_chunks = [
                    r["document"]
                    for r in self.rag_system.vector_db.similarity_search(question)
                ]

                # Calculate evaluation metrics
                faithfulness = self.evaluate_faithfulness(
                    question, rag_result["response"], retrieval_results["context"]
                )

                relevancy = self.evaluate_relevancy(question, rag_result["response"])

                context_precision = self.evaluate_context_precision(
                    question, retrieved_chunks
                )

                # Store evaluation result
                evaluation_results.append(
                    {
                        "question": question,
                        "answer": rag_result["response"],
                        "faithfulness": round(faithfulness, 3),
                        "relevancy": round(relevancy, 3),
                        "context_precision": round(context_precision, 3),
                        "response_time_sec": round(response_time, 2),
                        "vector_results": rag_result["vector_results_count"],
                        "web_results": rag_result["web_results_count"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            progress_bar.progress((i + 1) / len(test_questions))

        progress_bar.empty()

        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(evaluation_results)

        # Store in evaluation history
        self.evaluation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "results": evaluation_results,
                "summary": {
                    "avg_faithfulness": df["faithfulness"].mean(),
                    "avg_relevancy": df["relevancy"].mean(),
                    "avg_context_precision": df["context_precision"].mean(),
                    "avg_response_time": df["response_time_sec"].mean(),
                },
            }
        )

        return df

    def display_evaluation_results(self, df: pd.DataFrame):
        """Display evaluation results in Streamlit"""

        st.subheader("📊 RAG Evaluation Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Faithfulness",
                f"{df['faithfulness'].mean():.3f}",
                help="How well answers are grounded in retrieved context",
            )

        with col2:
            st.metric(
                "Relevancy",
                f"{df['relevancy'].mean():.3f}",
                help="How relevant answers are to the questions",
            )

        with col3:
            st.metric(
                "Context Precision",
                f"{df['context_precision'].mean():.3f}",
                help="How precise the retrieved context is",
            )

        with col4:
            st.metric(
                "Avg Response Time",
                f"{df['response_time_sec'].mean():.2f}s",
                help="Average time to generate responses",
            )

        # Detailed results table
        st.subheader("📋 Detailed Results")

        # Display results with expandable answers
        for i, row in df.iterrows():
            with st.expander(f"Question {i + 1}: {row['question'][:50]}..."):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write("**Question:**")
                    st.write(row["question"])

                    st.write("**Answer:**")
                    st.write(row["answer"])

                with col2:
                    st.write("**Metrics:**")
                    st.write(f"Faithfulness: {row['faithfulness']}")
                    st.write(f"Relevancy: {row['relevancy']}")
                    st.write(f"Context Precision: {row['context_precision']}")
                    st.write(f"Response Time: {row['response_time_sec']}s")
                    st.write(f"Vector Results: {row['vector_results']}")
                    st.write(f"Web Results: {row['web_results']}")

    def export_evaluation_results(self, df: pd.DataFrame) -> str:
        """Export evaluation results as JSON string"""
        return df.to_json(indent=2)

    def get_evaluation_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of latest evaluation"""
        if not self.evaluation_history:
            return None

        return self.evaluation_history[-1]["summary"]
