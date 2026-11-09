
import streamlit as st
from rag import ask_question


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Legal & Tax Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

.header-box {
    padding: 25px;
    border-radius: 14px;
    background: linear-gradient(120deg, #172554, #1e40af);
    color: white;
    margin-bottom: 25px;
}

.header-box h1 {
    color: white;
    font-size: 30px;
    margin-bottom: 8px;
}

.header-box p {
    color: #dbeafe;
    font-size: 15px;
}

.source-card {
    background-color: #f1f5f9;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 8px;
}

.small-text {
    color: #64748b;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =====================================================
# 4. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("⚖️ Legal Assistant")

    st.markdown("---")

    st.subheader("About")

    st.write(
        """
        Your AI-powered assistant for understanding
        Indian legal and tax compliance requirements.

        Ask questions about GST, business regulations,
        and legal provisions covered by the uploaded
        documents.
        """
    )

    st.markdown("---")

    st.subheader("Technology")

    st.markdown("""
    - **LLM:** Groq
    - **Model:** GPT-OSS 120B
    - **Embeddings:** Sentence Transformers
    - **Vector Database:** FAISS
    - **Frontend:** Streamlit
    """)

    st.markdown("---")

    st.subheader("Features")

    st.markdown("""
    - Document-based question answering
    - Semantic search
    - Source references
    - Context-aware responses
    """)

    st.markdown("---")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.caption(
        "Note: Responses are based on the available "
        "documents and should not replace professional "
        "legal or tax advice."
    )


# =====================================================
# 5. MAIN HEADER
# =====================================================

st.markdown("""
<div class="header-box">
    <h1>Indian Legal & Tax Compliance Assistant</h1>
    <p>
        Get answers to your legal and tax-related questions
        using AI-powered document retrieval.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 6. SUGGESTED QUESTIONS
# =====================================================

if not st.session_state.messages:

    st.subheader("How can I help you today?")

    st.write("Try asking one of these questions:")

    suggestions = [
        "Who is required to register under GST?",
        "What are the requirements for GST registration?",
        "What documents are needed for registration?",
        "What are the important MSME compliance requirements?"
    ]

    cols = st.columns(2)

    for i, question in enumerate(suggestions):

        with cols[i % 2]:

            if st.button(
                question,
                key=f"suggestion_{i}",
                use_container_width=True
            ):
                st.session_state.pending_question = question
                st.rerun()


# =====================================================
# 7. DISPLAY CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display sources for assistant messages
        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander(
                    f"📚 View Sources ({len(sources)} retrieved chunks)"
                ):

                    for i, source in enumerate(sources, start=1):

                        st.markdown(
                            f"**{i}. {source['document']}**"
                        )

                        st.caption(
                            f"Category: {source['category']} | "
                            f"Page: {source['page']} | "
                            f"Similarity: {source['similarity']}"
                        )

                        with st.container():

                            st.markdown(
                                f"<div class='source-card'>"
                                f"{source['text'][:1200]}"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                        st.markdown("---")


# =====================================================
# 8. PROCESS USER QUESTION
# =====================================================

question = st.chat_input(
    "Ask a question about Indian legal or tax compliance..."
)

# Handle suggested question clicks
if st.session_state.pending_question:

    question = st.session_state.pending_question
    st.session_state.pending_question = None


if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating your answer..."
        ):

            try:

                result = ask_question(question)

                answer = result.get(
                    "answer",
                    "No answer was generated."
                )

                sources = result.get("sources", [])

            except Exception as error:

                answer = (
                    "Something went wrong while processing "
                    "your question. Please try again."
                )

                sources = []

                st.error(str(error))

        st.markdown(answer)

        # Display retrieved sources
        if sources:

            with st.expander(
                f"📚 View Sources ({len(sources)} retrieved chunks)"
            ):

                for i, source in enumerate(sources, start=1):

                    st.markdown(
                        f"**{i}. {source['document']}**"
                    )

                    st.caption(
                        f"Category: {source['category']} | "
                        f"Page: {source['page']} | "
                        f"Similarity: {source['similarity']}"
                    )

                    st.markdown(
                        source["text"][:1200]
                    )

                    st.markdown("---")

        else:

            st.info("No supporting document chunks were retrieved.")

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    st.rerun()
