from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Function to get raw text from pdfs
def get_text(pdfs):
    """
    Takes in a list of pdfs and returns a string containing all the text in all pdfs appended one after the other
    """
    raw_text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            raw_text += page.extract_text()
        raw_text += """ 
                                                                              End of document.
        """
    return raw_text

# Function to break raw text into chunks
def get_chunks(raw_text, chunk_size=1000, chunk_overlap=100):
    """
    Takes in a large string of raw text and 
    returns the text as a list of strings, with some overlap"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(raw_text)
    return chunks