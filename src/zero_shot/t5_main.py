import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import TensorDataset, DistributedSampler, DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

sys.path.append(os.getcwd() + "/../../")  # noqa: E402

from src.common.gpu_utils import set_seed, init_gpu_params
from src.feature_utils.convert_examples_to_normal_prompt_features import convert_examples_to_t5_prompt_features
from src.common.arguments import ArgumentsConfig
from src.data_utils.processors import processors_mapping

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def _filter(output, end_token='<extra_id_1>'):
  _txt = t5_tokenizer.batch_decode(output[:, 2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)

  output = []

  for _ in _txt:
    if end_token in _:
      _end_token_idx = _.index(end_token)
      output.append(_[:_end_token_idx])
    else:
      output.append(_)

  return output


def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_path', type=str, required=True)
  parser.add_argument('--device_ids', type=str, default="0")
  parser.add_argument("--seed", type=int, default=13)
  parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
  parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs in the node.")
  return parser


if __name__ == '__main__':
  args = get_parser().parse_args()
  config = ArgumentsConfig.from_json_file(args.config_path)

  set_seed(args)
  init_gpu_params(args)

  Path(f'./t5_zero_shot_outputs/{config.task_name}').mkdir(parents=True, exist_ok=True)

  T5_PATH = 't5-large'  # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"
  t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)

  data_processor = processors_mapping[config.task_name]
  train_examples = data_processor.get_train_examples(config.dataset_dir)
  train_features = convert_examples_to_t5_prompt_features(train_examples,
                                                          t5_tokenizer,
                                                          config.task_name,
                                                          config.manual_template,
                                                          max_seq_length=config.max_seq_length,
                                                          lm_max_seq_length=config.max_seq_length + 10)
  train_dataset = TensorDataset(
    torch.tensor([f.input_ids for f in train_features], dtype=torch.long),
    torch.tensor([f.cls_label for f in train_features], dtype=torch.long)
  )

  train_sampler = DistributedSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.eval_batch_size,
                                pin_memory=True, num_workers=4)

  output_stores = defaultdict()
  data_labels = data_processor.get_labels()

  for i in data_labels:
    label = None
    if config.task_name == 'mnli' or config.task_name == 'snli':
      if i == 'contradiction':
        label = 0
      elif i == 'entailment':
        label = 1
      elif i == 'neutral':
        label = 2
      else:
        raise ValueError(f'Wrong label mapping: {i}')
    elif config.task_name == 'qnli' or config.task_name == 'rte':
      if i == 'entailment':
        label = 0
      elif i == 'not_entailment':
        label = 1
      else:
        raise ValueError(f'Wrong label mapping: {i}')
    else:
      label = int(i)

    output_stores[label] = {}

  logger.info(output_stores)

  t5_config = T5Config.from_pretrained(T5_PATH)
  t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config)
  t5_mlm.eval()
  if config.is_cuda:
    t5_mlm.cuda()

  model = DistributedDataParallel(t5_mlm, device_ids=[args.device_ids[args.local_rank]],
                                  output_device=args.device_ids[args.local_rank], find_unused_parameters=True)

  pbar = tqdm(train_dataloader, desc='Iter', disable=not args.is_master)

  for idx, batch in enumerate(pbar):
    if args.multi_gpu:
      train_sampler.set_epoch(idx * 1000)
    outputs = model.module.generate(input_ids=batch[0].cuda(),
                                    num_beams=30, num_return_sequences=15,
                                    no_repeat_ngram_size=2,
                                    max_length=10)

    dist.barrier()
    cls_label = batch[1].detach().cpu().tolist()
    results = _filter(outputs)

    for r, l in zip(results, cls_label):
      if r not in output_stores[l]:
        output_stores[l][r] = 1
      else:
        output_stores[l][r] += 1

    pbar.set_postfix({'rank': args.local_rank})

  for k, v in output_stores.items():
    output_stores[k] = sorted(output_stores[k].items(), key=lambda x: x[1], reverse=True)

  with open(f'./t5_zero_shot_outputs/{config.task_name}/{config.task_name}-train-zeroshot-rank{args.local_rank}.json',
            'w', encoding='utf-8') as f:
    json.dump(output_stores, f, ensure_ascii=False, indent=2)
