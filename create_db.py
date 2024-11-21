from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder

import os
import streamlit as st

import nltk

def create_pinecone_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    nltk.download('punkt_tab')
    
    index_name = "hybrid-search-langchain-pinecone"

    # initialize Pinecone client
    pc = Pinecone(api_key=st.session_state["PINECONE_API_KEY"])

    # create the index
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # dimensionality of dense model - we are going to use Hugging Face embeddings
            metric="dotproduct",  # sparse values supported only for dotproduct
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)

    # use default tf-idf values
    bm25_encoder = BM25Encoder().default()

    # fit tf-idf values on your corpus
    bm25_encoder.fit(chunks)

    # store the values to a json file
    bm25_encoder.dump("bm25_values.json")

    # load to your BM25Encoder object
    bm25_encoder = BM25Encoder().load("bm25_values.json")

    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

    retriever.add_texts(chunks)

    return retriever