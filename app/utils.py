from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    T5ForConditionalGeneration, T5Tokenizer
)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_bart_model(model_path="harshvardhini123/fine_tuned_bart"):
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    return tokenizer, model

def load_flan_t5_lora(base_model_name="t5-small", adapter_path="harshvardhini123/fine_tuned_model"):
    model = T5ForConditionalGeneration.from_pretrained(adapter_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(base_model_name)
    return tokenizer, model

def generate_summary(model, tokenizer, text, max_input_length=512, max_output_length=150):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_input_length, truncation=True).to(device)
    summary_ids = model.generate(
        **inputs,
        max_length=max_output_length,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def ensemble_summary(input_text, bart_model, bart_tokenizer, t5_model, t5_tokenizer, alpha=0.6):
    summary_bart = generate_summary(bart_model, bart_tokenizer, input_text)
    summary_t5 = generate_summary(t5_model, t5_tokenizer, input_text)

    # Simple weighted ensemble: pick longer or sample probabilistically
    if len(summary_bart.split()) > len(summary_t5.split()):
        return summary_bart
    else:
        return summary_t5 if torch.rand(1).item() > alpha else summary_bart
