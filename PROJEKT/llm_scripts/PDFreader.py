from langchain.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Initialize the Ollama model
ollama = Ollama(base_url='http://localhost:11434', model="LLAMA2_22_11_512")

# Load the PDF document using PyPDFLoader
pdf_loader = PyPDFLoader("/home/user/mashinlerning/Machine_Learning_for_Robust_Network_Design_A_New_Perspective.pdf")
data = pdf_loader.load()

# Split the documents using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
all_splits = text_splitter.split_documents(data)

# Create OllamaEmbeddings
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="LLAMA2_22_11_512")

# Create Chroma vectorstore from documents
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

# Define the question for QA
question = "write a 20 page seminar based on the text provided."

# Perform similarity search for the question
docs = vectorstore.similarity_search(question)
print(f"Similar documents found: {len(docs)}")

# Create RetrievalQA chain
qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())

# Get the answer for the question
result = qachain({"query": question})
print(result)
