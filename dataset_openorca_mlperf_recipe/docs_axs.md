## Usage

This recipe downloads the OpenOrca dataset used for Llama2-70b inference submissions

### Command Anatomy
Only run the 2nd producer rule.

```bash
axs byquery downloaded,dataset_name=openorca,
            model_family=llama2, # The model family to use - Note more changes will need to be made for this to properly work with different model families. The reason for this is that the input column has llama2 specific tags, which must be converted.
            variant=7b,
            total_samples=24576 # The total number of samples to convert
```
### Misc
Make sure you've downloaded the relevant tokeniser first. 