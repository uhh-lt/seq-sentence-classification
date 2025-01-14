# Ollama

We use ollama to host our models.

## Model customization

Models are customized (context size is increased) with the help of modelfiles.
We use A100 GPU with 80GB VRAM for all experiments.

```
docker compose exec ollama bash
ollama create -f /modelfiles/my-gemma-2 my_gemma_2:latest
ollama create -f /modelfiles/my-llama-31 my_llama_31:latest
ollama create -f /modelfiles/my-mistral-nemo my_mistral_nemo:latest
```
