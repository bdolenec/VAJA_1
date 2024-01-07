from langchain.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Inicializacija OLLAMA modela
ollama = Ollama(base_url='http://localhost:11434', model="LLAMA2_22_11_512")

# Naloži pdf dokument
pdf_loader = PyPDFLoader("/home/user/mashinlerning/Machine_Learning_for_Robust_Network_Design_A_New_Perspective.pdf")
data = pdf_loader.load()

# Zaradi dolžine teksta, tega dokumenta nemoremo naenkrat sprocesirati, zato tekst presekamo na več delov
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
all_splits = text_splitter.split_documents(data)

oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="LLAMA2_22_11_512")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

# Uprašanja za model na podlagi danega teksta
question = "what is a GAT network"

# Iskanje podobnosti
docs = vectorstore.similarity_search(question)
print(f"Similar documents found: {len(docs)}")
qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())

# Dobljen rezultat ki ga izda LLM
result = qachain({"query": question})
print(result)
