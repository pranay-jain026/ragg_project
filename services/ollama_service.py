import requests
from langchain_ollama import ChatOllama


def get_llm(model, base_url):
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0,
    )


def check_ollama_health(base_url, chat_model, embed_model):
    try:
        res = requests.get(f"{base_url}/api/tags", timeout=5)
        res.raise_for_status()

        models = res.json().get("models", [])
        names = [m["name"].split(":")[0] for m in models]

        missing = [m for m in [chat_model, embed_model] if m not in names]

        if missing:
            return False, f"Missing models: {missing}"

        return True, "Ollama is ready"

    except Exception as e:
        return False, str(e)