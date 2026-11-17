# Models under test (Chapter 3 table; verify quantisation with `ollama show <tag>`)

| Key | Ollama tag | Params | Base / family | Training data (headline) | Quant | Licence |
|---|---|---|---|---|---|---|
| llama3 | llama3:8b | 8B | Meta Llama 3 Instruct | 15T tokens public web, instruction tuned | Q4_0 (Ollama default) | Llama 3 Community |
| mistral | mistral:7b | 7.3B | Mistral 7B Instruct v0.3 | undisclosed web corpus, instruction tuned | Q4_0 | Apache 2.0 |
| qwen | qwen2.5:7b | 7.6B | Qwen2.5 Instruct | 18T tokens multilingual, instruction tuned | Q4_K_M | Apache 2.0 |
| gemma | gemma:7b | 8.5B | Google Gemma 1 Instruct | 6T tokens web, code, maths | Q4_0 | Gemma Terms |
| meditron | meditron:7b | 7B | Llama-2-7B continued pretraining | PubMed, guidelines, GAP-Replay (EPFL) | Q4_0 | Llama 2 Community |
| biomistral | hf.co/BioMistral/BioMistral-7B-GGUF:Q4_K_M | 7.3B | Mistral-7B-Instruct-v0.1 continued pretraining | PubMed Central open access | Q4_K_M | Apache 2.0 |

Medical fine-tuning pairs: BioMistral vs mistral:7b (clean pair, same base). Meditron vs llama3:8b is NOT a clean
pair (Meditron is Llama-2-based); report it as such.
