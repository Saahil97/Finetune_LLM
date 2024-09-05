# Mistral WWE GGUF Model

Welcome to the Mistral WWE GGUF model repository! This model has been fine-tuned on WWE-related trivia data and is designed to generate multiple-choice questions and answers in the context of WWE events, superstars, and history.

## Model Details

- **Model Name**: Mistral WWE GGUF (https://huggingface.co/Saahil97/mistral_wwetrivia)
- **Model Architecture**: [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- **Model Type**: Causal Language Model
- **Training Data**: Fine-tuned on a custom dataset of WWE-related trivia.
- **Format**: GGUF (General Graph Universal Format)

## Usage

### Installation

To use this model, you will need the `transformers` and `llama_cpp` libraries installed. You can install them using pip:

```bash
pip install transformers llama_cpp 
```

```bash
pip install -r requirements.txt
```

### Training the Model

The model was fine-tuned on a custom dataset consisting of WWE trivia questions. The fine-tuning process involved training the model for 10 epochs using a batch size of 16 on a single GPU.
Training is done on Mistral-7B-Instruct-v0.1,finetunnig is done using the dataset provided in the repo, dataset comprises of 200 distinct questions and answers.

```bash
python wweMCQ_train.py
```

After fine-tuning, merging of the base Mistral model with any specialized LoRA adjustments is streamlined through a single command, enhancing the model's capabilities with minimal additional parameters.

```bash
python merge.py --base_model /../../.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/2dcff66eac0c01dc50e4c41eea959968232187fe --lora_model /../../../../../finetunellm/mistral-7b-wwetuned --output_type huggingface --output_dir dir_name
```
### Inferencing the Model

Inferencing can be done directly using the model or by calling the API file which inturn calls the inference file

```bash
python api.py
```

```bash
from transformers import AutoTokenizer
from llama_cpp import Llama

model_path = "path/to/gguf_model"

model = Llama(model_path=model_path)

input_text = "Who won the first Royal Rumble in 1988?"

# Generate output
output = model(input_text, max_tokens=8000)  
result = output['choices'][0]

text = result['text']
print(text)
```

### Output Format

```bash
Question: Who won the first Royal Rumble in 1988?
Options:
a: Jim Duggan
b: Hulk Hogan
c: Randy Savage
d: Andre the Giant
Correct Answer: a: Jim Duggan
```

## Hyperparameters
- **Learning Rate**: 2e-4
- **Batch Size**: 16
- **Number of Epochs**: 10
- **Optimizer**: AdamW

### Model Performance

The model has been evaluated on a validation set of WWE trivia questions, achieving an accuracy of 95% on multiple-choice questions.

### Limitations and Biases

- **Domain Specific**: The model is fine-tuned specifically for WWE-related trivia and may not perform well on general knowledge questions.
- **Bias**: The model may reflect biases present in the training data.

### Citation
```bibtex
@misc{mistral_wwe_trivia_2024,
  author = {Saahil97},
  title = {Mistral WWE GGUF Model},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/Saahil97/mistral_wwetrivia}
}
```
