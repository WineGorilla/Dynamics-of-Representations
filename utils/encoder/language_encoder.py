from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np

def load_model(model_name,device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name,output_hidden_states=True)
    model.to(device)
    model.eval()
    return tokenizer,model



def get_text_embeddings(words,device,batch_size,tokenizer,model):
    all_layers = []
    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i + batch_size]

            inputs = tokenizer(
                batch_words,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32
            ).to(device)

            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # (layers, batch, seq, dim)

            cls_layers = [h[:, 0, :].cpu().numpy() for h in hidden_states]

            if not all_layers:
                all_layers = [x for x in cls_layers]
            else:
                for li in range(len(cls_layers)):
                    all_layers[li] = np.concatenate(
                        [all_layers[li], cls_layers[li]], axis=0
                    )

    return np.stack(all_layers, axis=0)  # (n_layers, n_words, dim)