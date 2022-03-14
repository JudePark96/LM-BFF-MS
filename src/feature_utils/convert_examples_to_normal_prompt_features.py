import itertools
import logging
from typing import List

import torch
from tqdm import tqdm
from transformers import InputExample, PreTrainedTokenizer

from src.feature_utils.input_features_definer import BasePromptInputFeatures
from src.verbalizers.verbalizer_definer import Verbalizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_examples_to_prompt_features(examples: List[InputExample],
                                        tokenizer: PreTrainedTokenizer,
                                        task: str,
                                        manual_template: str = None,
                                        max_seq_length: int = 256,
                                        lm_max_seq_length: int = 512) -> List[BasePromptInputFeatures]:
  """
  :param examples:
  :param tokenizer:
  :param task:
  :param manual_template: It was [MASK], ? [MASK] , ...
  :param max_seq_length:
  :param lm_max_seq_length:
  :return:
  """
  features = []

  if manual_template is None:
    logger.info(f'Manual Template is None, Null-Prompting Mode')
  else:
    logger.info(f'Manual Template is: {manual_template}')

  label_space = [Verbalizer[v] for v in list(filter(lambda x: x == task, list(Verbalizer)))][0]
  label_space = list(itertools.chain(*[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(l)) for l in label_space]))

  logger.info(label_space)

  if tokenizer.mask_token_id not in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(manual_template)):
    raise ValueError('Wrong template! Template should contain mask_token_id!')

  for example in tqdm(examples, total=len(examples)):
    text_a = example.text_a
    text_b = example.text_b
    text_a = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_a))

    if text_b is not None:
      text_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_b))

    if text_b is None:
      inputs = torch.tensor(
        [tokenizer.bos_token_id] +
        text_a[:max_seq_length] +
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(manual_template)) +
        [tokenizer.eos_token_id]
      ).long()

      masked_position = (inputs == tokenizer.mask_token_id).nonzero()
      label = int(example.label)
      mlm_label = torch.full((lm_max_seq_length,), -100).long()
      mlm_label[masked_position] = label_space[label]

      input_ids = torch.full((lm_max_seq_length,), tokenizer.pad_token_id)
      input_ids[:inputs.shape[-1]] = inputs

      features.append(BasePromptInputFeatures(
        input_ids=input_ids.tolist(),
        attention_mask=input_ids.ne(tokenizer.pad_token_id).float().tolist(),
        mlm_label=mlm_label.tolist(),
        cls_label=label
      ))
    else:
      inputs = torch.tensor(
        [tokenizer.bos_token_id] +
        (text_a +
         tokenizer.convert_tokens_to_ids(tokenizer.tokenize(manual_template)) +
         text_b)[:max_seq_length] +
        [tokenizer.eos_token_id]).long()
      masked_position = (inputs == tokenizer.mask_token_id).nonzero()

      label = None
      if task == 'mnli' or task == 'snli':
        if example.label == 'contradiction':
          label = 0
        elif example.label == 'entailment':
          label = 1
        elif example.label == 'neutral':
          label = 2
        else:
          raise ValueError(f'Wrong label: {example.label}')
      elif task == 'qnli' or task == 'rte':
        if example.label == 'entailment':
          label = 0
        elif example.label == 'not_entailment':
          label = 1
        else:
          raise ValueError(f'Wrong label: {example.label}')
      else:
        label = int(example.label)

      mlm_label = torch.full((lm_max_seq_length,), -100).long()
      mlm_label[masked_position] = label_space[label]

      input_ids = torch.full((lm_max_seq_length,), tokenizer.pad_token_id)
      input_ids[:inputs.shape[-1]] = inputs

      features.append(BasePromptInputFeatures(
        input_ids=input_ids.tolist(),
        attention_mask=input_ids.ne(tokenizer.pad_token_id).float().tolist(),
        mlm_label=mlm_label.tolist(),
        cls_label=label
      ))

  return features


def convert_examples_to_t5_prompt_features(examples: List[InputExample],
                                           tokenizer: PreTrainedTokenizer,
                                           task: str,
                                           manual_template: str = None,
                                           max_seq_length: int = 256,
                                           lm_max_seq_length: int = 512) -> List[BasePromptInputFeatures]:
  """
  :param examples:
  :param tokenizer:
  :param task:
  :param manual_template: It was [MASK], ? [MASK] , ...
  :param max_seq_length:
  :param lm_max_seq_length:
  :return:
  """
  features = []

  if manual_template is None:
    logger.info(f'Manual Template is None, Null-Prompting Mode')
  else:
    logger.info(f'Manual Template is: {manual_template}')

  # label_space = [Verbalizer[v] for v in list(filter(lambda x: x == task, list(Verbalizer)))][0]
  # label_space = list(itertools.chain(*[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(l)) for l in label_space]))
  #
  # logger.info(label_space)

  for example in tqdm(examples, total=len(examples)):
    text_a = example.text_a
    text_b = example.text_b
    text_a = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_a))

    if text_b is not None:
      text_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_b))

    if text_b is None:
      inputs = torch.tensor(
        text_a[:max_seq_length] +
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(manual_template)) +
        [tokenizer.eos_token_id]
      ).long()

      label = int(example.label)

      input_ids = torch.full((lm_max_seq_length,), tokenizer.pad_token_id)
      input_ids[:inputs.shape[-1]] = inputs

      features.append(BasePromptInputFeatures(
        input_ids=input_ids.tolist(),
        cls_label=label
      ))
    else:
      inputs = torch.tensor(
        (text_a +
         tokenizer.convert_tokens_to_ids(tokenizer.tokenize(manual_template)) +
         text_b)[:max_seq_length] +
        [tokenizer.eos_token_id]).long()

      label = None
      if task == 'mnli' or task == 'snli':
        if example.label == 'contradiction':
          label = 0
        elif example.label == 'entailment':
          label = 1
        elif example.label == 'neutral':
          label = 2
        else:
          raise ValueError(f'Wrong label: {example.label}')
      elif task == 'qnli' or task == 'rte':
        if example.label == 'entailment':
          label = 0
        elif example.label == 'not_entailment':
          label = 1
        else:
          raise ValueError(f'Wrong label: {example.label}')
      else:
        label = int(example.label)

      input_ids = torch.full((lm_max_seq_length,), tokenizer.pad_token_id)
      input_ids[:inputs.shape[-1]] = inputs

      features.append(BasePromptInputFeatures(
        input_ids=input_ids.tolist(),
        cls_label=label
      ))

  return features
