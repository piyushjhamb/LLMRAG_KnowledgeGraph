import streamlit as st
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import EnsembleRetriever
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import chromadb.api
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# Clear ChromaDB cache
chromadb.api.client.SharedSystemClient.clear_system_cache()

# ============ Streamlit UI Setup ============
st.title("Hybrid Knowledge Graph RAG System")
st.write("Upload PDF files to create a knowledge graph with hybrid retrieval (Graph + Vector Search)")

# Configuration Section
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

# Neo4j Configuration
st.sidebar.subheader("Neo4j Settings")
neo4j_url = st.sidebar.text_input("Neo4j URL", value="neo4j://localhost:7687")
neo4j_user = st.sidebar.text_input("Neo4j Username", value="neo4j")
neo4j_password = st.sidebar.text_input("Neo4j Password", type="password", value="password")

# File upload constraints
MAX_FILES = 5
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

if not api_key:
    st.warning("Please enter your Groq API Key in the sidebar")
    st.stop()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="Gemma2-9b-It",
    temperature=0
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.info(f"Upload up to {MAX_FILES} PDF files (max {MAX_FILE_SIZE_MB}MB each)")

# File uploader
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
    help=f"Upload up to {MAX_FILES} PDF files, each up to {MAX_FILE_SIZE_MB}MB"
)

# Validate file uploads
if uploaded_files:
    if len(uploaded_files) > MAX_FILES:
        st.error(f"Error: You can upload a maximum of {MAX_FILES} files. You uploaded {len(uploaded_files)} files.")
        st.stop()

    # Check file sizes
    oversized_files = []
    for uploaded_file in uploaded_files:
        file_size = uploaded_file.size
        if file_size > MAX_FILE_SIZE_BYTES:
            oversized_files.append((uploaded_file.name, file_size / (1024 * 1024)))

    if oversized_files:
        st.error("The following files exceed the 10MB size limit:")
        for filename, size_mb in oversized_files:
            st.error(f"  - {filename}: {size_mb:.2f}MB")
        st.stop()

    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

    # Process PDFs
    documents = []
    with st.spinner("Processing PDF files..."):
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp_{uploaded_file.name}"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

            # Clean up temp file
            os.remove(temp_pdf)

            st.write(f"Loaded: {uploaded_file.name} ({len(docs)} pages)")

    st.success(f"Total {len(documents)} pages loaded from all PDFs")

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    st.info(f"Split into {len(texts)} text chunks")

    # ============ Neo4j Graph Setup ============
    with st.spinner("Setting up Neo4j Graph Store..."):
        try:
            graph = Neo4jGraph(
                url=neo4j_url,
                username=neo4j_user,
                password=neo4j_password
            )
            st.success("Connected to Neo4j!")
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {e}")
            st.info("Make sure Neo4j is running on localhost:7687")
            st.stop()

    # ============ Vector Similarity Retrieval Setup (Chroma) ============
    with st.spinner("Setting up Vector Similarity Retrieval (Chroma)..."):
        vectorstore_chroma = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            collection_name="knowledge_graph_hybrid_pdf"
        )

        chroma_retriever = vectorstore_chroma.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        st.success("Chroma vector store created!")

    # ============ Neo4j Vector Retrieval Setup ============
    with st.spinner("Setting up Neo4j Vector Index..."):
        try:
            # Clear existing vector index data
            graph.query("MATCH (n:TextChunk) DETACH DELETE n")

            neo4j_vector = Neo4jVector.from_documents(
                texts,
                embeddings,
                url=neo4j_url,
                username=neo4j_user,
                password=neo4j_password,
                index_name="pdf_text_embeddings",
                node_label="TextChunk"
            )

            neo4j_retriever = neo4j_vector.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            st.success("Neo4j vector index created!")
        except Exception as e:
            st.error(f"Error creating Neo4j vector index: {e}")
            st.stop()

    # ============ Hybrid Retrieval - Ensemble Method ============
    st.subheader("Hybrid Retrieval Configuration")

    col1, col2 = st.columns(2)
    with col1:
        chroma_weight = st.slider("Chroma Vector Weight", 0.0, 1.0, 0.5, 0.1)
    with col2:
        neo4j_weight = st.slider("Neo4j Vector Weight", 0.0, 1.0, 0.5, 0.1)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, neo4j_retriever],
        weights=[chroma_weight, neo4j_weight],
    )

    st.success("Hybrid Retrieval System Ready!")
    st.info(f"Configuration: Chroma ({chroma_weight*100:.0f}%) + Neo4j ({neo4j_weight*100:.0f}%)")

    # ============ Query Interface ============
    st.divider()
    st.subheader("Query Your Documents")

    # Query input
    user_query = st.text_input("Enter your question:", placeholder="e.g., What are the main topics in the documents?")

    # Query type selection
    query_type = st.radio(
        "Select retrieval method:",
        ["Hybrid (Recommended)", "Vector Search Only", "Graph Query (Cypher)"],
        horizontal=True
    )

    if user_query:
        if query_type == "Hybrid (Recommended)":
            with st.spinner("Searching with hybrid retrieval..."):
                results = hybrid_retriever.invoke(user_query)

                st.write(f"**Found {len(results)} relevant documents:**")
                for i, doc in enumerate(results, 1):
                    with st.expander(f"Result {i} - Source: {doc.metadata.get('source', 'Unknown')}"):
                        st.write(doc.page_content)
                        st.caption(f"Page: {doc.metadata.get('page', 'N/A')}")

        elif query_type == "Vector Search Only":
            with st.spinner("Searching with vector similarity..."):
                results = chroma_retriever.invoke(user_query)

                st.write(f"**Found {len(results)} relevant documents:**")
                for i, doc in enumerate(results, 1):
                    with st.expander(f"Result {i} - Source: {doc.metadata.get('source', 'Unknown')}"):
                        st.write(doc.page_content)
                        st.caption(f"Page: {doc.metadata.get('page', 'N/A')}")

        elif query_type == "Graph Query (Cypher)":
            with st.spinner("Querying Neo4j graph..."):
                try:
                    cypher_chain = GraphCypherQAChain.from_llm(
                        llm=llm,
                        graph=graph,
                        verbose=True
                    )

                    graph_result = cypher_chain.invoke({"query": user_query})
                    st.write("**Graph Query Result:**")
                    st.success(graph_result['result'])

                    with st.expander("View Generated Cypher Query"):
                        st.code(graph_result.get('cypher', 'N/A'), language="cypher")
                except Exception as e:
                    st.error(f"Graph query failed: {e}")
                    st.info("Try using 'Hybrid' or 'Vector Search Only' instead")

    # ============ Statistics ============
    st.divider()
    st.subheader("System Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", len(documents))
    with col2:
        st.metric("Text Chunks", len(texts))
    with col3:
        st.metric("Embedding Dimensions", 384)

else:
    st.info("Please upload PDF files to get started")
    st.markdown("""
    ### Features:
    - Upload up to 5 PDF files (max 10MB each)
    - Hybrid retrieval combining vector search + graph context
    - Dual vector stores: Chroma + Neo4j
    - Powered by Groq LLM (Gemma2-9b-It)
    - Configurable retrieval weights
    """)