#Uvoz potrebnih modulov iz knjižnice LANGCHAIN
from langchain.llms import Ollama
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain

# Ustvarjanje instance WebBaseLoader za nalaganje besedilnih dokumentov iz spletnega vira
loader = WebBaseLoader("https://en.wikipedia.org/wiki/2023_Plateau_State_massacres")
# Nalaganje besedil iz podanega spletnega vira
docs = loader.load()

# Ustvarjanje instance Ollama za uporabo določenega modela
llm = Ollama(model="LLAMA2_22_11_512")
#Nalaganje verige za povzemanje besedil s pomočjo modela Ollama
chain = load_summarize_chain(llm, chain_type="stuff")

# Zagon verige povzemanja na naloženih dokumentih
result = chain.run(docs)
# Izpis rezultata
print(result)