
# Jezikovni Modeli v Praksi: Modificiranje in uporaba velikih jezikovnih modelov z OLLAMA

![AI](robot.jpg)

## Delovanje Nevronskih Omrežij (Neural Networks)

Nevronska omrežja so računalniški modeli, katerih inspiracija so bili možgani. Temeljijo na vse-povezanosti velikega števila nevronov, ki so povezani med seboj prek uteženih povezav. Te uteži se prilagajajo med učenjem, da omogočajo omrežju, da se prilagaja in uči na podlagi vhodnih podatkov.

## Delovanje LLM (Large Language Models)

Veliki jezikovni modeli (LLM) so zmogljivi modeli umetne inteligence, ki so usposobljeni za razumevanje in generiranje človeškega jezika. Ti modeli temeljijo na nevronskih mrežah, zlasti na transformacijskih nevronskih mrežah (Transformer). LLM-i so trenirani na velikih količinah besedilnih podatkov, kar jim omogoča, da ustvarjajo besedila, prevajajo jezike, odgovarjajo na vprašanja in še veliko več.

## Delovanje LLM v povezavi s PyTorch, TensorFlow in CUDA

PyTorch in TensorFlow sta dve najbolj priljubljeni knjižnici za strojno učenje, ki omogočata enostavno ustvarjanje in upravljanje nevronskih omrežij, vključno z LLM-ji. Obe knjižnici zagotavljata orodja za izgradnjo, treniranje in uporabo teh modelov.

CUDA je platforma podjetja Nvidia, ki omogoča pospeševanje izvajanja računalniških operacij z uporabo grafičnih procesnih enot (GPU). PyTorch in TensorFlow podpirata CUDA, kar omogoča hitrejše izvajanje operacij nevronskih omrežij na grafičnih karticah Nvidia, kar bistveno pospeši učenje in delovanje velikih modelov, kot so LLM-ji.

# **Navodila za namestitev in pripravo okolja**

Tu bom opisal kako sem sam prišel do rezultatov na mojem sistemu Windows 11. Za ostale sisteme isti postopki ne veljajo zaradi različnosti hardvera in operacisjkih sistemov

## Omogočanje Funkcije WSL

Wsl(windows subsystem for linux) je funkcionalnost windows operacijskega sistema ki omogoča poganjanje različnih linux distribucij lokalno v windowsih. več si lahko preberete na tej strani: *https://learn.microsoft.com/en-us/windows/wsl/about*

 - potrebujete windows 10 ali 11
 - zaženite powershell kot administrator in vpišite komando: 
 ```bash
 wsl --install
 ```
 - ko je inštalacije konec resetirajte sistem
 - obstajata dve verziji, wsl2 je novejša:
```bash
wsl --set-default-version 2 
```
 - win + R in vpišite optional features 
 - v optional features obkljukajte windows subsystem for linux, Hypervisor, Hyper-V, Virtual machine platform
 - izpiše trenutne linux distribucije ki so omogočene za inštalacijo:
```bash
wsl --list --online
```
 - inštaliramo željeno distribucijo, priporočeno: ubuntu
```bash
wsl --install -d Ubuntu
```
 - preveri delovanje wsl:
```bash
wsl -l -v
```
 - **'wsl --shutdown'** ugasne wsl **'wsl.exe'** požene wsl. **'wsl --help'** za več pomoči
 - *https://learn.microsoft.com/en-us/windows/wsl/install* za dodatno pomoč

## Nvidia CUDA

V tem koraku omogočimo deljenje resursov nvidia grafične kartice okolju WSL Ubuntu z Nvidia CUDA in CUDA Tool kit

 - Inštalacija cude za windows in wsl: *https://docs.nvidia.com/cuda/wsl-user-guide/index.html*
 - CUDA toolkit za Ubuntu Wsl:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3
```
 - za dodatne informacije: *https://developer.nvidia.com/cuda-downloads*
 - za preverjanje inštalacije:
```bash
nvidia-smi
nvcc --version
```
 - debug:
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## OLLAMA

Sedaj lahko inštaliramo ollama. Ollama je terminalna aplikacija ki vključuje vse zahtevane knjižnice ki so potrebne za zagon LLM-ov lokalno, in poenostavi interakcijo, inštaliranje, in modificiranje LLM-ov

 - inštalira ollama:
```bash
curl https://ollama.ai/install.sh | sh
```
 - zažene proces ki se izvaja lokalno in je serviran na localhost:11434 po defoltu:
```bash
ollama serve
```
 - pomoč za ollama:
```bash
Usage:
ollama [flags]
ollama [command]

Available Commands:
serve       Start ollama
create      Create a model from a Modelfile
show        Show information for a model
run         Run a model
pull        Pull a model from a registry
push        Push a model to a registry
list        List models
cp          Copy a model
rm          Remove a model
help        Help about any command

Flags:
-h, --help      help for ollama
-v, --version   Show version information
```
## Uporaba ollama

Sedaj lahko začnemo uporabljati ollama

 - potegnemo želen model in ga zaženemo. za listo modelov: *https://ollama.ai/library* 
```bash
ollama run <model>
```
 - ollama podpira tudi docker. kako uporabiti docker: *https://ollama.ai/blog/ollama-is-now-available-as-an-official-docker-image* 
```bash
docker exec -it ollama ollama run <model>
```
 - če je vse prav se bo izbran model zagnal v terminalu.
 - več o ollama: *https://github.com/jmorganca/ollama/tree/main* , *https://ollama.ai* 

## modificiranje modelov

Modele lahko modificiramo in upravljamo z količino resorsov ki jih model lahko uporablja.

to naredimo z pomočjo MODELFILE

 - v visual studio code najprej dodamo add-on wsl, nato z klikom na spodnji levi gumb [><] pokonektamo wsl in visual studio
 - v novi mapi naredimo datoteko modelfile
 - primer model fila: 
```python
FROM llama2:latest #The FROM instruction defines the base model to use when creating a model.

TEMPLATE """[INST] <<SYS>>{{ .System }}<</SYS>>
{{ .Prompt }} [/INST]
"""

SYSTEM """<system message>""" # The SYSTEM instruction specifies the system message to be used in the template

PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
PARAMETER stop "<<SYS>>"
PARAMETER stop "<</SYS>>"
PARAMETER num_gpu 22 #The number of layers to send to the GPU(s). 0 > only runs on CPU
PARAMETER num_ctx 512 #Sets the size of the context window used to generate the next token
PARAMETER num_thread 12 #Sets the number of threads to use during computation
```
 - več o modelfiles: *https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md*
 - modelfile shranimo in v terminal napišemo sledeče:
```bash
ollama create <izbrano ime za nov model> -f ./modelfile
```
 - da poženemo nov modificiran model: 
```bash
ollama run <ime novega modela>
```

# Python skripte in Langchain

O Langchain si lahko preberete tukaj: *https://www.langchain.com*

Tu je nekaj skript kjer sem skombiniral Langchain knjižnice in ollama za delanje povzetkov pdf dokumentov, spletnih strani in spletni vmesnik za lažjo interakcijo ki ni terminalska s knjižnico GrAdio

Z inštalacijo Langchaina se inštalira zraven tudi veliko 'python dependecies' zato predlagam delovanje v virtualnem okolju:
 - potegnemo skript za kreiranje virtualnih okolij:
```bash
pip install venv
```
 - naredimo virtualno okolje:
```bash
python3 -m venv env
```
 - aktiviramo virtualno okolje:
```bash
source env/bin/activate
```
 - potegnemo langchain:
```bash
pip install langchain
```
 - če imamo težavo z dependencies mormao izvesti še ta korak:
```bash
pip -r install requirements.txt
```

### bralec ki povzame spletne strani:

```python
# Uvoz potrebnih modulov iz knjižnice LANGCHAIN
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
```

### bralec spletnih strani katerega lahko vprašaš vsebinska uprašanja

```python
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
```

### Bralec pdf dokumentov, ki jih povzame:

```python
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
```
### spletni grafični vmesnik (web gui)

```python
# Uvoz potrebnih knjižnic
import requests # Za pošiljanje HTTP zahtevkov
import json # Za delo s JSON podatki
import gradio as gr # Za ustvarjanje vmesnika

# URL, ki vodi do strežnika z LLM-om(do našega porta na katerem ponujamo storitev)
url = "http://localhost:11434/api/generate"

# Definicija zaglavij za HTTP zahtevke
headers = {
    'Content-Type': 'application/json',
}

# Seznam za shranjevanje zgodovine pogovora
conversation_history = []

# Funkcija, ki generira odziv na podlagi podanega spodbujevalnega besedila
def generate_response(prompt):
    conversation_history.append(prompt) #dodajanje v zgodovino

    full_prompt = "\n".join(conversation_history) #združitev zgodovine
    # Pripravi podatke za zahtevek na strežnik z LLM-om
    data = {
"model": "LLAMA2_22_11_512", # Določi model
"stream": False,
"prompt": full_prompt, # Celotno zgodovino pogovora uporabi kot spodbudo
    }
    # Pošlji POST zahtevek na določen URL s pripravljenimi podatki in zaglavji
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # Preveri, ali je bil odgovor uspešen (status 200)
    if response.status_code == 200:
response_text = response.text
data = json.loads(response_text)
actual_response = data["response"] # Pridobi generiran odgovor iz prejetih podatkov
conversation_history.append(actual_response) # Dodaj odgovor v zgodovino pogovora
return actual_response # Vrni generiran odgovor
    else:
print("Error:", response.status_code, response.text) # V primeru napake izpiši podrobnosti
return None # Vrnemo None v primeru napake

# Ustvari vmesnik Gradio
iface = gr.Interface(
    fn=generate_response, # Uporabi funkcijo generate_response kot osnovno funkcijo vmesnika
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."), # Vhodno polje za vnos besedila
    outputs="text" # Prikaži izhodni tekst
)
# Zaženi vmesnik
iface.launch()
```

vse skripte lahko poženemo z komando 
```bash
python3 <ime datoteke s kodo>
```
