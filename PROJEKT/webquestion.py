from langchain.llms import Ollama
ollama = Ollama(base_url='http://localhost:11434',
model="LLAMA2_22_11_512")

from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikipedia.org/wiki/2023_Plateau_State_massacres")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
all_splits = text_splitter.split_documents(data)

from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="LLAMA2_22_11_512")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

question="how many people died and how many were injured?"
docs = vectorstore.similarity_search(question)
len(docs)

from langchain.chains import RetrievalQA
qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
result = qachain({"query": question})
print(result)