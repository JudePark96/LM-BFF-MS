import itertools
import logging
import random
from copy import copy
from typing import List

import torch
from tqdm import tqdm
from transformers import InputExample, PreTrainedTokenizer

from src.feature_utils.input_features_definer import PromptWithContinuousDemonstrationFeatures
from src.verbalizers.verbalizer_definer import Verbalizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_examples_to_conti_demon_prompt_features(examples: List[InputExample],
                                                    tokenizer: PreTrainedTokenizer,
                                                    task: str,
                                                    demonstration_tokens: List[List[str]],
                                                    soft_token_length: int,
                                                    soft_token: str = '[T]',
                                                    manual_template: str = None,
                                                    max_seq_length: int = 256,
                                                    demonstration_mode: str = 'static',  # static or ranking
                                                    lm_max_seq_length: int = 512,
                                                    is_training: bool = True) -> List[
  PromptWithContinuousDemonstrationFeatures]:
  """
  :param demonstration_mode:
  :param soft_token:
  :param soft_token_length:
  :param demonstration_tokens:
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

  if demonstration_mode == 'static':
    total_soft_tokens = [s for s in
                         range(1, len(list(itertools.chain(*copy(demonstration_tokens)))) * soft_token_length + 1)]
  else:
    total_soft_tokens = [s for s in
                         range(1, 5 * soft_token_length + 1)]

  if tokenizer.mask_token_id not in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(manual_template)):
    logger.info(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(manual_template)))
    logger.info(tokenizer.mask_token_id)
    raise ValueError('Wrong template! Template should contain mask_token_id!')

  for idx, example in enumerate(tqdm(examples, total=len(examples))):
    text_a = example.text_a
    text_b = example.text_b
    text_a = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_a))

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

    demonstration = []

    if demonstration_mode == 'static':
      for tokens in demonstration_tokens:
        if text_b is None:
          demon_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(soft_token)) * soft_token_length +
                       tokenizer.convert_tokens_to_ids(
                         ' '.join(tokenizer.tokenize(manual_template)).replace(tokenizer.mask_token,
                                                                               ' '.join(tokenizer.tokenize(d_t))).split(
                           ' '))
                       for d_t in tokens] + [[tokenizer.eos_token_id]]
        else:
          demon_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(soft_token)) * (soft_token_length // 2) +
                       tokenizer.convert_tokens_to_ids(
                         ' '.join(tokenizer.tokenize(manual_template)).replace(tokenizer.mask_token,
                                                                               ' '.join(tokenizer.tokenize(d_t))).split(
                           ' ')) +
                       tokenizer.convert_tokens_to_ids(tokenizer.tokenize(soft_token)) * (soft_token_length // 2)
                       for d_t in tokens] + [[tokenizer.eos_token_id]]
        demonstration.append(demon_ids)
      demonstration = list(itertools.chain(*list(itertools.chain(*demonstration))))
    else:
      # We set maximum token count of demonstration is 5
      if is_training:
        filtered_demonstration_tokens = [i[0] for i in demonstration_tokens[idx]][:5]
      else:
        sampled_demon_tok_idx = random.randint(1, len(demonstration_tokens) - 1)
        filtered_demonstration_tokens = [i[0] for i in demonstration_tokens[sampled_demon_tok_idx]][:5]

      for tokens in filtered_demonstration_tokens:
        if text_b is None:
          demon_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(soft_token)) * soft_token_length + \
                      tokenizer.convert_tokens_to_ids(
                        ' '.join(tokenizer.tokenize(manual_template))
                          .replace(tokenizer.mask_token, ' '.join(tokenizer.tokenize(tokens))).split(' ')
                      ) + [tokenizer.eos_token_id]
        else:
          demon_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(soft_token)) * (soft_token_length // 2) + \
                      tokenizer.convert_tokens_to_ids(
                        ' '.join(tokenizer.tokenize(manual_template))
                          .replace(tokenizer.mask_token, ' '.join(tokenizer.tokenize(tokens))).split(' ')
                      ) + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(soft_token)) * (soft_token_length // 2) + \
                      [tokenizer.eos_token_id]

        demonstration.append(demon_ids)
      demonstration = list(itertools.chain(*demonstration))

    if idx == 0:
      logger.info(f'demonstration => {tokenizer.decode(demonstration)}')

    if text_b is not None:
      text_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_b))

    if text_b is None:
      inputs = torch.tensor(
        [tokenizer.bos_token_id] +
        text_a[:max_seq_length] +
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(manual_template)) +
        [tokenizer.eos_token_id]
      ).long()
    else:
      inputs = torch.tensor(
        [tokenizer.bos_token_id] +
        text_a[:max_seq_length] +
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(manual_template)) +
        text_b[:max_seq_length] +
        [tokenizer.eos_token_id]).long()
      # inputs = torch.tensor(
      #   [tokenizer.bos_token_id] +
      #   text_a[:max_seq_length] +
      #   text_b[:max_seq_length] +
      #   tokenizer.convert_tokens_to_ids(tokenizer.tokenize(manual_template)) +
      #   [tokenizer.eos_token_id]).long()

    inputs = torch.cat((inputs, torch.tensor(demonstration).long()), dim=-1)
    if idx == 0:
      logger.info(f'inputs => {tokenizer.decode(inputs.tolist())}')
    demonstration_soft_input_ids = torch.tensor(total_soft_tokens).long()

    masked_position = (inputs == tokenizer.mask_token_id).nonzero()
    mlm_label = torch.full((lm_max_seq_length,), -100).long()
    mlm_label[masked_position] = label_space[label]

    input_ids = torch.full((lm_max_seq_length,), tokenizer.pad_token_id)
    input_ids[:inputs.shape[-1]] = inputs

    demonstration_soft_label = torch.full(input_ids.shape, 0)
    demonstration_soft_label += input_ids.eq(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(soft_token))[0])

    features.append(PromptWithContinuousDemonstrationFeatures(
      input_ids=input_ids.tolist(),
      attention_mask=input_ids.ne(tokenizer.pad_token_id).float().tolist(),
      demonstration_soft_input_ids=demonstration_soft_input_ids.tolist(),
      demonstration_soft_label=demonstration_soft_label.tolist(),
      mlm_label=mlm_label.tolist(),
      cls_label=label
    ))

  return features
