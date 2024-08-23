import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
from pathlib import Path
from tqdm import tqdm

from swift.llm import (
    get_model_tokenizer, get_template, inference,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch
import yaml


model_type = "internvl2-8b"
model_path = 'checkpoints/InternVL2-8B'
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')
model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_kwargs={'device_map': 'auto'},
                                       model_id_or_path=model_path)

# for GPUs that do not support flash attention
# model, tokenizer = get_model_tokenizer(model_type, torch.float16,
#                                        model_kwargs={'device_map': 'auto'},
#                                        model_id_or_path=model_path,
#                                        use_flash_attn = False)

model.generation_config.max_new_tokens = 2048
template = get_template(template_type, tokenizer)
seed_everything(42)



# PROMPT_PATH = Path('datasets/nlp_dataset/prompt')
# PROMPT_DICT = {}
# EVALUATION_PROMPT_DICT = {}
TRAIN_IMAGE_PATH = Path('Train')
VAL_IMAGE_PATH = Path('Val')

"""
def get_prompts():
    for root, dirs, files in os.walk(PROMPT_PATH):
        for file in files:
            if not file.endswith('.txt'):
                continue
            key = Path(root).relative_to(PROMPT_PATH) / file.replace('.txt', '')
            key = str(key)
            if file != 'test.txt':
                with open(Path(root) / file, 'r') as f:
                    PROMPT_DICT[key] = f.read()
            elif file == 'test.txt':
                with open(Path(root) / file, 'r') as f:
                    EVALUATION_PROMPT_DICT[key] = f.read()
"""
    
def llm_infer(img_path: Path, query: str):
    images = [str(img_path)]
    response, history = inference(model, template, query, images=images)
    return response

"""
def generate_dataset(dataset_path: Path, num_images: int = -1):
    result_dict = {}
    if num_images != -1:
        img_path_list = list(dataset_path.iterdir())
        img_path_list = random.sample(img_path_list, num_images)
    else:
        img_path_list = list(dataset_path.iterdir())

    pbar = tqdm(img_path_list, total=len(img_path_list))
    for img_path in pbar:
        one_img_result = {}
        for key in PROMPT_DICT.keys():
            pbar.set_description(f'img: {img_path.name}, prompt: {key}')
            result = llm_infer(img_path, PROMPT_DICT[key])
            '''
            if "prompt_en" in key:
                eval_prompt = EVALUATION_PROMPT_DICT['prompt_en/test']
            elif "prompt_zh" in key:
                eval_prompt = EVALUATION_PROMPT_DICT['prompt_zh/test']
            else:
                raise ValueError(f"test prompt for key: {key} no found.")
            score = llm_infer(img_path, eval_prompt + result)

            one_img_result[key] = {
                'result': result,
                'score': score
            }
            '''
            one_img_result[key] = result
        result_dict[str(img_path.name)] = one_img_result
    return result_dict
"""
    
def generate_dataset_simple(dataset_path: Path, num_images: int = -1):
    result_dict = {}
    if num_images != -1:
        img_path_list = list(dataset_path.iterdir())
        img_path_list = random.sample(img_path_list, num_images)
    else:
        img_path_list = list(dataset_path.iterdir())

    pbar = tqdm(img_path_list, total=len(img_path_list))
    for img_path in pbar:
        one_img_result = {}
        prompt = "请简要描述这张图片"
        pbar.set_description(f'img: {img_path.name}, prompt: {prompt}')
        result = llm_infer(img_path, prompt)
        one_img_result[prompt] = result
        result_dict[str(img_path.name)] = one_img_result
    return result_dict

if __name__ == '__main__':
    # get_prompts()
    result_dict = generate_dataset_simple(TRAIN_IMAGE_PATH)
    # os.makedirs('./results', exist_ok=True)
    with open('nlp_train.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(result_dict, f, allow_unicode=True)
    result_dict = generate_dataset_simple(VAL_IMAGE_PATH)
    with open('nlp_val.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(result_dict, f, allow_unicode=True)
