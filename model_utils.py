import gc
import time

import torch
from config import MODEL_PATH
from image_utils import load_image_tensor
from PIL import Image
from transformers import AutoModel, AutoTokenizer


def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # torch_dtype = torch.float16
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    model = (
        AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, device, torch_dtype


def describe_tree(image_pil, model, tokenizer, device, torch_dtype):
    resized = image_pil.resize((448, 448), Image.Resampling.BICUBIC)
    pixel_values = load_image_tensor(resized).to(dtype=torch_dtype, device=device)
    question = (
        "<image>\n"
        "Определи, есть ли на изображении хотя бы одно дерево или кустарник.\n"
        "Если НЕТ — ответь строго: 'Нет дерева на изображении'.\n"
        "Если ЕСТЬ — выбери только ОДИН объект.\n"
        "Дай ответ строго в формате:\n"
        "Вид и порода: <название породы или 'Неизвестно'>\n"
        "а на новой строке: Тип: <Дерево, Кустарник или Пень>\n\n"
        "Определяя тип:\n"
        "- Дерево — это Большое растение с главным стволом и могут отходить от него много других стволов, обычно выше 1 м.\n"
        "- Кустарник — маленький куст от земли\n"
        "- Пень - обрубок дерева\n"
        "Если сомневаешься, оцени высоту и наличие главного ствола и выбери наиболее вероятный вариант.\n"
        "Если думаешь, что это кустарник, то поясни, почему это кустарник"
        # "Не добавляй других комментариев и описаний."
    )

    start = time.time()
    with torch.no_grad():
        resp = model.chat(
            tokenizer,
            pixel_values,
            question,
            dict(
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            ),
        )
    return resp.strip(), time.time() - start
