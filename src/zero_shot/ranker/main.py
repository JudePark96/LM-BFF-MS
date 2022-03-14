import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import torch
from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, InputExample

sys.path.append(os.getcwd() + "/../../../")  # noqa: E402

from src.common.arguments import ArgumentsConfig
from src.common.label_utils import convert_str_to_discrete_label
from src.data_utils.processors import processors_mapping

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
  config, zero_shot_prediction_path, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
  args = ArgumentsConfig.from_json_file(config)

  Path(f'./ranker_output/{args.task_name}/').mkdir(parents=True, exist_ok=True)

  with open(zero_shot_prediction_path, 'rb') as f:
    zero_shot_prediction = json.load(f)

  tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-large')
  model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-large')
  model.eval()

  data_processor = processors_mapping[args.task_name]
  label_count = len(data_processor.get_labels())
  train_examples: List[InputExample] = data_processor.get_train_examples(args.dataset_dir)

  outputs = []

  for example in tqdm(train_examples):
    label = convert_str_to_discrete_label(args.task_name, example.label)
    text = example.text_a

    if example.text_b is not None:
      text += ' ' + example.text_b

    # stores = []

    # for c in range(label_count):
    texts = [text]
    texts.extend([i[0] for i in zero_shot_prediction])

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
      embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    semantic_stores = [(texts[i], (1 - cosine(embeddings[0], embeddings[i]))) for i in range(1, len(texts))]
    semantic_stores = sorted(semantic_stores, key=lambda x: x[1], reverse=True)[:10]

    outputs.append(semantic_stores)

  with open(f'./ranker_output/{args.task_name}/{args.random_seed}-rank-output.json', 'w', encoding='utf-8') as f:
    json.dump(outputs, f, ensure_ascii=False, indent=2)
