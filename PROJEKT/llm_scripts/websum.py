from langchain.llms import Ollama
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain

loader = WebBaseLoader("https://en.wikipedia.org/wiki/2023_Plateau_State_massacres")
docs = loader.load()

llm = Ollama(model="LLAMA2_22_11_512")
chain = load_summarize_chain(llm, chain_type="stuff")

result = chain.run(docs)
print(result)