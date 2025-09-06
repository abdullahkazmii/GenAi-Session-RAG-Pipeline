import streamlit as st
from datetime import datetime
from typing import List

from config import validate_config, OPENAI_API_KEY, SERPER_API_KEY
from rag_system import RAGSystem
from rag_evaluation import RAGEvaluator
from document_processor import DocumentProcessor


st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []
if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False


def initialize_system():
    """Initialize RAG system with environment variables"""
    try:
        is_valid, openai_key, serper_key = validate_config()

        if not is_valid:
            return False

        if st.session_state.rag_system is None:
            st.session_state.rag_system = RAGSystem()

        if st.session_state.rag_system.initialize(openai_key, serper_key):
            st.session_state.system_initialized = True
            return True
        else:
            st.error("Failed to initialize RAG system")
            return False

    except Exception as e:
        st.error(f"Error during system initialization: {str(e)}")
        return False


def main():
    """Main application"""

    st.title("End-to-End RAG with ChromaDB Vector Database")

    if not st.session_state.system_initialized:
        with st.spinner("Initializing RAG system..."):
            if not initialize_system():
                st.error(
                    "Failed to initialize system. Please check your .env file contains valid API keys."
                )
                st.info("""
                Create a `.env` file in your project directory with:
                ```
                OPENAI_API_KEY=your_openai_api_key_here
                SERPER_API_KEY=your_serper_api_key_here
                ```
                """)
                return

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["\t Chat", "\t Knowledge Base", "\t Web Search"])

    with tab1:
        chat_interface()

    with tab2:
        knowledge_base_interface()

    with tab3:
        web_search_interface()

    # with tab4:
    #     evaluation_interface()


def chat_interface():
    st.header("Chat with Your Knowledge Base")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Ask questions about your uploaded documents or general topics")

    chat_container = st.container(height=400)

    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message
        with chat_container.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with chat_container.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rag_result = st.session_state.rag_system.generate_response(prompt)

                st.markdown(rag_result["response"])

        # Add assistant response
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": rag_result["response"],
                "sources": rag_result["sources"],
            }
        )

        st.rerun()


def knowledge_base_interface():
    """Interface for managing knowledge base documents"""
    # Show current stats
    if st.session_state.rag_system:
        stats = st.session_state.rag_system.get_system_stats()

        col1, col2 = st.columns(2)

        with col1:
            st.header("Knowledge Base Management")
        with col2:
            # if st.button("🗑️ Clear Knowledge Base"):
            #     if st.session_state.rag_system.clear_knowledge_base():
            #         st.session_state.processed_documents = []
            #         st.rerun()
            pass

    # Document upload interface
    doc_tab1, doc_tab2 = st.tabs(["\t PDF Upload", "\t Websites"])

    with doc_tab1:
        st.subheader("Upload PDF Documents")

        pdf_files = st.file_uploader(
            "Choose PDF files", type=["pdf"], accept_multiple_files=True
        )

        if pdf_files and st.button("Process PDFs"):
            for pdf_file in pdf_files:
                with st.spinner(f"Processing {pdf_file.name}..."):
                    text = DocumentProcessor.extract_pdf_text(pdf_file)
                    if text:
                        success = st.session_state.rag_system.add_document(
                            text, "PDF", pdf_file.name
                        )
                        if success:
                            st.session_state.processed_documents.append(
                                {
                                    "name": pdf_file.name,
                                    "type": "PDF",
                                    "timestamp": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M"
                                    ),
                                }
                            )

    with doc_tab2:
        st.subheader("Website Content")

        website_urls = st.text_area(
            "Website URLs (one per line)",
            placeholder="https://example.com\nhttps://blog.example.com",
        )

        if website_urls and st.button("Process Websites"):
            urls = [url.strip() for url in website_urls.split("\n") if url.strip()]

            for url in urls:
                with st.spinner(f"Processing {url}..."):
                    result = DocumentProcessor.extract_website_content(url)
                    if result:
                        text, title = result
                        success = st.session_state.rag_system.add_document(
                            text, "Website", title
                        )
                        if success:
                            st.session_state.processed_documents.append(
                                {
                                    "name": title,
                                    "type": "Website",
                                    "timestamp": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M"
                                    ),
                                }
                            )


def web_search_interface():
    """Web search interface"""

    st.header("Web Search")

    # Manual search
    search_query = st.text_input("Search from web", placeholder="Enter search query...")

    col1, col2 = st.columns([3, 1])
    with col2:
        num_results = st.selectbox("Results", [3, 5, 10], index=1)

    if search_query and st.button("Search"):
        with st.spinner("Searching..."):
            results = st.session_state.rag_system.web_searcher.search(
                search_query, num_results
            )

            if results:
                st.subheader("Search Results")
                for i, result in enumerate(results, 1):
                    st.write(f"**{i}. {result['title']}**")
                    st.write(result["snippet"])
                    st.write(f"🔗 [Read more]({result['link']})")
                    st.divider()
            else:
                st.info("No results found")


# def evaluation_interface():
#     """RAG evaluation interface using RAGAS concepts"""

#     st.header("RAG Evaluation (RAGAS)")

#     st.info("""
#     **RAGAS (RAG Assessment)** evaluates RAG system performance using:
#     - **Faithfulness**: How well answers are grounded in retrieved context
#     - **Answer Relevancy**: How relevant answers are to questions
#     - **Context Precision**: How precise the retrieved context is
#     - **Context Recall**: How complete the retrieved context is
#     """)

#     # Create evaluator
#     evaluator = RAGEvaluator(st.session_state.rag_system)

#     (eval_tab1,) = st.tabs(["Run Evaluation"])

#     with eval_tab1:
#         st.subheader("Test Questions")

#         custom_questions = st.text_area(
#             "Enter test questions (one per line)",
#             height=200,
#             placeholder="What is the main topic?\nHow does X relate to Y?\nWhat are the benefits of Z?",
#         )

#         # Run evaluation
#         questions_to_evaluate = []

#         if hasattr(st.session_state, "test_questions"):
#             questions_to_evaluate.extend(st.session_state.test_questions[:3])

#         if custom_questions:
#             custom_q_list = [
#                 q.strip() for q in custom_questions.split("\n") if q.strip()
#             ]
#             questions_to_evaluate.extend(custom_q_list[:5])  # Limit for demo

#         if questions_to_evaluate:
#             st.write(f"**Questions to evaluate:** {len(questions_to_evaluate)}")

#             if st.button("🚀 Run RAG Evaluation", type="primary"):
#                 with st.spinner("Running RAGAS evaluation..."):
#                     # Run evaluation
#                     eval_df = evaluator.run_evaluation(questions_to_evaluate)

#                     # Display results
#                     evaluator.display_evaluation_results(eval_df)

#                     # Store results in session state
#                     st.session_state.latest_evaluation = eval_df
#         else:
#             st.info("Create test questions above to run evaluation")


if __name__ == "__main__":
    main()
