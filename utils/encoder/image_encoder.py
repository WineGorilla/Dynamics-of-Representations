import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np
from tqdm import tqdm


def load_image_model(model_name="facebook/dinov2-base", device="mps"):
    print(f"加载图像模型: {model_name}")

    try:
        extractor = AutoImageProcessor.from_pretrained(model_name)
    except Exception:
        extractor = AutoFeatureExtractor.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()
    return extractor, model


# 计算embedding
def get_image_embeddings(extractor, model, image_paths, device, all_layers=True, cls_only=True, batch_size=4):
    all_hidden_layers = []
    model.eval()

    # 分批处理
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = extractor(images=batch_imgs, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  

        if cls_only:
            layer_embeds = [h[:, 0, :].detach() for h in hidden_states]
        else:
            layer_embeds = [h.mean(dim=1).detach() for h in hidden_states]

        all_hidden_layers.append(layer_embeds)

    n_layers = len(all_hidden_layers[0])  # 每个样本层数相同
    # 拼接每层的所有 batch
    layer_concat = [torch.cat([batch[i] for batch in all_hidden_layers], dim=0) for i in range(n_layers)]

    # 转 numpy
    X_layers = [layer.cpu().numpy() for layer in layer_concat]

    # 输出与语言模型格式一致
    return X_layers





