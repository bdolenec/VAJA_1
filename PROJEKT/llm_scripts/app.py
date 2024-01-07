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