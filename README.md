# DistilBERT LoRA Text Classification Model

A fine-tuned DistilBERT model for text classification using LoRA (Low-Rank Adaptation) technique. This model provides efficient parameter-efficient fine-tuning while maintaining high performance.

## Model Description

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Task**: Text Classification
- **Checkpoint**: 25,000 training steps

## Requirements

```bash
pip install torch transformers peft
```

## Quick Start

### Loading the Model

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftConfig, PeftModel

# Path to your fine-tuned model
peft_path = "Sent-Bert\kaggle_notebook_outputs\my_model"

# Load configuration
config = PeftConfig.from_pretrained(peft_path)

# Load base model
model_base = AutoModelForSequenceClassification.from_pretrained(
    config.base_model_name_or_path,
    local_files_only=True,
    use_safetensors=True
)

# Load LoRA weights
model = PeftModel.from_pretrained(model_base, peft_path)

# Load tokenizer (can use local tokenizer files or base model)
tokenizer = AutoTokenizer.from_pretrained(peft_path)
# Alternative: tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
```

### Inspecting Model Configuration

```python
# View LoRA configuration
print("LoRA Configuration:")
print(f"- Base model: {config.base_model_name_or_path}")
print(f"- Task type: {config.task_type}")
print(f"- LoRA rank: {config.r}")
print(f"- LoRA alpha: {config.lora_alpha}")
print(f"- Target modules: {config.target_modules}")

# View model info
print(f"\nModel has {model.num_parameters():,} total parameters")
print(f"Trainable parameters: {model.get_nb_trainable_parameters():,}")
```


## Model Architecture

This model uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning:

- **Base Model**: DistilBERT with 67M parameters
- **LoRA Parameters**: Only a small fraction of parameters are trainable
- **Memory Efficient**: Significantly reduced memory footprint compared to full fine-tuning
- **Performance**: Maintains competitive performance with full fine-tuning


**Key Files:**
- `adapter_config.json`: Contains LoRA configuration parameters
- `adapter_model.safetensors`: The actual fine-tuned LoRA weights
- `tokenizer.json` & `vocab.txt`: Tokenizer files for text preprocessing
- `training_args.bin`: Records the training parameters used

## Performance Considerations

### GPU Usage
```python
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# For inference
result = classify_text(text, model, tokenizer, device=device)
```

### Memory Optimization
```python
# Enable memory efficient attention (if using newer PyTorch)
torch.backends.cuda.enable_flash_sdp(True)

# Use half precision for inference (if using GPU)
if device.type == 'cuda':
    model = model.half()
```

### Complete Working Example

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftConfig, PeftModel

# Setup
peft_path = "Sent-Bert\kaggle_notebook_outputs\my_model"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and tokenizer
config = PeftConfig.from_pretrained(peft_path)
model_base = AutoModelForSequenceClassification.from_pretrained(
    config.base_model_name_or_path,
    local_files_only=True,
    use_safetensors=True
)
model = PeftModel.from_pretrained(model_base, peft_path)
tokenizer = AutoTokenizer.from_pretrained(peft_path)

# Define your class labels (customize these)
id2label = {
    0: "Negative",
    1: "Positive"
}

# Move model to device
model.to(device)

# Example texts
text_list = [
    "This product is amazing!",
    "I hate this service.",
    "The movie was okay."
]

# Make predictions
print("Trained model predictions:")
print("--------------------------")

for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(inputs).logits
    
    predictions = torch.argmax(logits, dim=1)
    print(text + " - " + id2label[predictions.item()])
```

## Training Information

- **Training Steps**: 25,000
- **Base Model**: distilbert-base-uncased
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Framework**: Hugging Face Transformers + PEFT
- **Year**: 2025


## Citation

If you use this model in your research, please cite:

```bibtex
@misc{patel2025distilbert-lora,
  author = {Jinit Patel},
  title = {DistilBERT LoRA Text Classification},
  year = {2025},
  url = {https://github.com/porcupine1829/sent-bert}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate and follow the existing code style.

## Contact

**Jinit Patel**
- GitHub: [@porcupine1829](https://github.com/porcupine1829)
- Email: [jenis1829@gmail.com]

For questions about this model or collaboration opportunities, feel free to reach out!
