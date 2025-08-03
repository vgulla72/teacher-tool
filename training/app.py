import ollama

def model_prompt(model_name, prompt):
    print(prompt)
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "system",
                   "content": "You are a knowledgeable and eloquent teacher of Indian epics, specifically the Ramayana and Mahabharata. Your responses are drawn from texts and respected commentaries, and you aim to provide clear, respectful, and culturally grounded answers. Explain concisely using simple language."
                              "Explain as if you're telling this to children in a classroom"},
                    {"role": "user", "content": prompt}]
    )

    return response['message']['content'].strip()

if __name__ == "__main__":
    prompt = "Name all pandavasa?"
    response = model_prompt("tinyllama:chat", prompt)
    print("Response from base model")
    print(response)
    response = model_prompt("tinyllama-epic", prompt)
    print("Response from trained model")
    print(response)