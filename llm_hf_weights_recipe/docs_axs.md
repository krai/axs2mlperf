## Usage

You can use this recipe to download the weights or tokeniser for an LLM on Huggingface.

### Command anatomy
```bash
axs byquery downloaded,
            hf_tokeniser, # download a the tokeniser, use `hf_model` to download the model instead
            hf_token=$HF_TOKEN, # The token used to access the model, can be obtained from Huggingface
            model_family=llama2, # The model family, e.g. llama2 or mixtral
            variant=7b # The model variant
```