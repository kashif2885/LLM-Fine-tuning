from flask import Flask, render_template
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from tqdm.auto import tqdm

app=Flask(__name__)



@app.route("/custom_model",methods=['POST'])
def custom_model():
    # Define local path and load model and tokenizer
    local_model_path = "fine-tuned-model"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = '''
    <|im_start|>system
    You are an AI assistant to guide people about constitution of Pakistan. You should understand the context of user and need to provide a useful information
    <|im_end|>

    <|im_start|>user
    in which year the constitution of Pakistan was updated?<|im_end|>

    <|im_start|>assistant

    '''

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=1000,
            num_return_sequences=1,
            do_sample=True
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    return generated_text

if __name__=="__main__":
    app.run(debug=True)