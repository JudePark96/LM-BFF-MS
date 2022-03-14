__author__ = 'Eunhwan Jude Park'
__email__ = 'judepark@{jbnu.ac.kr, kookmin.ac.kr}'
__repository__ = 'https://github.com/JudePark96'

import itertools
import json
import logging
import math
import os
import random
import sys
from abc import ABC
from copy import copy
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from transformers import PreTrainedTokenizer, AdamW, get_linear_schedule_with_warmup, RobertaForMaskedLM, \
  RobertaTokenizer, AutoModelForMaskedLM, AutoTokenizer

from src.common.arguments import ArgumentsConfig
from src.data_utils.add_soft_prompt_tokens import add_special_tokens_and_resize_embedding
from src.data_utils.processors import processors_mapping, compute_metrics_mapping
from src.feature_utils.convert_examples_to_conti_demon_prompt_features import \
  convert_examples_to_conti_demon_prompt_features
from src.feature_utils.input_features_definer import PromptWithContinuousDemonstrationFeatures
from src.modeling.continuous_demonstration_prompt_modeling import ContiDemonPromptModel
from src.verbalizers.verbalizer_definer import Verbalizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTrainer:
  def __init__(self,
               args: ArgumentsConfig) -> None:
    super().__init__()
    self.args = args

  def get_dataloader(self,
                     examples: List[PromptWithContinuousDemonstrationFeatures],
                     tokenizer: PreTrainedTokenizer,
                     soft_token: str,
                     demonstration_tokens: List[List[str]],
                     demonstration_mode: str,
                     is_training: bool = True):
    features = convert_examples_to_conti_demon_prompt_features(examples,
                                                               tokenizer,
                                                               task=self.args.task_name,
                                                               soft_token=soft_token,
                                                               soft_token_length=self.args.soft_token_length,
                                                               demonstration_tokens=demonstration_tokens,
                                                               demonstration_mode=demonstration_mode,
                                                               manual_template=self.args.manual_template,
                                                               max_seq_length=self.args.max_seq_length,
                                                               lm_max_seq_length=512,
                                                               is_training=is_training)
    dataset = TensorDataset(
      torch.tensor([f.input_ids for f in features], dtype=torch.long),
      torch.tensor([f.attention_mask for f in features], dtype=torch.float),
      torch.tensor([f.demonstration_soft_input_ids for f in features], dtype=torch.long),
      torch.tensor([f.demonstration_soft_label for f in features], dtype=torch.long),
      torch.tensor([f.mlm_label for f in features], dtype=torch.long),
      torch.tensor([f.cls_label for f in features], dtype=torch.long)
    )

    dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True) if is_training \
      else DataLoader(dataset, batch_size=self.args.eval_batch_size, pin_memory=True, num_workers=4)
    return dataset, dataloader

  def count_parameters(self, model: nn.Parameter):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

  def save_model_state_dict(self, save_state_dict_dir, state_dict_file, model):
    """
    Save model state dict. Note this function is not for reproduce or continuing train.

    Arguments:
        model (nn.Module):
        save_state_dict_dir (str):
        state_dict_file (str):
    """

    model_to_save = model.module if hasattr(model, 'module') else model
    save_path = os.path.join(save_state_dict_dir, state_dict_file)
    torch.save(model_to_save.state_dict(), save_path, _use_new_zipfile_serialization=False)
    logger.info(f'{save_path} model saved!')

  def get_verbalizer(self, task_name: str, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    label_space = [Verbalizer[v] for v in list(filter(lambda x: x == task_name, list(Verbalizer)))][0]
    label_space = list(itertools.chain(*[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(l)) for l in label_space]))
    return torch.tensor(label_space)

  def train(self):
    raise NotImplementedError()

  def eval(self, loader: DataLoader, mode: str = 'Dev'):
    raise NotImplementedError()

  # This is for analysis
  def inference(self):
    raise NotImplementedError()


class DemonstrationPromptTrainer(BaseTrainer):
  def __init__(self, args: ArgumentsConfig) -> None:
    super().__init__(args)
    data_processor = processors_mapping[args.task_name]
    train_examples = data_processor.get_train_examples(args.dataset_dir)
    dev_examples = data_processor.get_dev_examples(args.dataset_dir)
    test_examples = data_processor.get_test_examples(args.dataset_dir)

    self.base = RobertaForMaskedLM.from_pretrained(args.lm_model)
    self.tokenizer = RobertaTokenizer.from_pretrained(args.lm_model)

    logger.info(f'AutoModelForMaskedLM, AutoTokenizer: {args.lm_model} loaded!')

    soft_tokens = add_special_tokens_and_resize_embedding(self.tokenizer, self.base)

    self.model = ContiDemonPromptModel(self.args, self.base,
                                       total_soft_len=len(list(itertools.chain(
                                         *copy(self.args.demonstration_tokens)))) * self.args.soft_token_length,
                                       verbalizer_idx=self.get_verbalizer(self.args.task_name, self.tokenizer).tolist())

    if args.demonstration_rank_output_path is not None:
      with open(args.demonstration_rank_output_path, 'rb') as f:
        demonstrations = json.load(f)
        mode = 'rank'
    else:
      demonstrations = args.demonstration_tokens
      mode = 'static'

    logger.info(f'demonstration tokens of {self.args.task_name} -> {demonstrations}')

    self.train_dataset, self.train_dataloader = self.get_dataloader(train_examples,
                                                                    self.tokenizer,
                                                                    soft_tokens,
                                                                    demonstrations,
                                                                    mode)
    self.dev_dataset, self.dev_dataloader = self.get_dataloader(dev_examples, self.tokenizer, soft_tokens,
                                                                demonstrations, mode, False)
    self.test_dataset, self.test_dataloader = self.get_dataloader(test_examples, self.tokenizer,
                                                                  soft_tokens, demonstrations, mode, False)

    self.total_train_step = len(
      self.train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
    self.warm_steps = math.ceil(self.total_train_step * self.args.warmup_proportion)

    parameters = [
      {'params': [p for p in self.model.parameters() if p.requires_grad]}
    ]

    self.optimizer = AdamW(parameters, lr=args.learning_rate)
    self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warm_steps,
                                                     num_training_steps=self.total_train_step)

    logger.info(
      f'Total number of trainable parameters: {self.count_parameters(self.model)}')

    if self.args.is_cuda:
      logger.info('Now all parameter is in CUDA option.')
      self.model.cuda()
      self.verbalizer = self.get_verbalizer(self.args.task_name, self.tokenizer).cuda()
    self.writer = SummaryWriter(log_dir=self.args.tensorboard_output_dir)

    logger.info('Zero-Shot Test Evaluation')
    self.global_steps = 0
    self.early_stop_epoch = 0
    # self.eval(self.dev_dataloader, 'Zero-Shot Test')

  def train(self):
    train_iterator = trange(int(self.args.num_train_epochs), desc='Epoch')
    self.model.train()
    self.model.zero_grad()

    self.test_best_acc = 0.0
    self.dev_best_acc = 0.0

    # Only for MRPC, QQP.
    self.test_best_f1 = 0.0
    self.dev_best_f1 = 0.0

    for _ in train_iterator:
      epoch_iterator = tqdm(self.train_dataloader, desc='Iteration')
      iter_loss = 0.0

      for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.cuda() for t in batch) if self.args.is_cuda else batch
        model_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'demonstration_soft_input_ids': batch[2],
                        'demonstration_soft_label': batch[3],
                        'mlm_label': batch[4],
                        'cls_label': batch[5],
                        'verbalizer': self.verbalizer,
                        'is_eval': False}
        model_output = self.model(**model_inputs)
        loss = model_output['loss']

        if self.args.gradient_accumulation_steps > 1:
          loss /= self.args.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % self.args.gradient_accumulation_steps == 0:
          if self.args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

          self.optimizer.step()
          self.scheduler.step()
          self.model.zero_grad()
          self.global_steps += 1

        iter_loss += loss.item()
        epoch_iterator.set_postfix({
          "epoch": f"{_}",
          "global_steps": f"{self.global_steps}",
          "prompt_learning_rate": f"{self.scheduler.get_last_lr()[0]:.10f}",
          "rolling_loss": f"{iter_loss / (step + 1) * self.args.gradient_accumulation_steps:.5f}",
          "last_loss": f"{loss.item() * self.args.gradient_accumulation_steps:.5f}"
        })
      dev_output = self.eval(self.dev_dataloader, mode='Dev')

      if self.args.task_name not in ['qqp', 'mrpc']:
        if 'mnli' in self.args.task_name:
          dev_acc = dev_output['mnli/acc']
        else:
          dev_acc = dev_output['acc']
        if dev_acc >= self.dev_best_acc:
          self.dev_best_acc = dev_acc
          test_output = self.eval(self.test_dataloader, mode='Test')
          if 'mnli' in self.args.task_name:
            test_acc = test_output['mnli/acc']
          else:
            test_acc = test_output['acc']
          if test_acc > self.test_best_acc:
            self.test_best_acc = test_acc
            self.save_model_state_dict(self.args.tensorboard_output_dir, 'best_model.pth', self.model)
      else:
        dev_f1 = dev_output['f1']
        if dev_f1 >= self.dev_best_f1:
          self.dev_best_f1 = dev_f1
          test_output = self.eval(self.test_dataloader, mode='Test')
          test_f1 = test_output['f1']
          if test_f1 > self.test_best_f1:
            self.test_best_f1 = test_f1
            self.save_model_state_dict(self.args.tensorboard_output_dir, 'best_model.pth', self.model)

  def eval(self, loader: DataLoader, mode: str = 'Dev'):
    self.model.eval()

    preds = []
    trues = []

    for batch in tqdm(loader, desc=f'{mode} Iteration'):
      batch = tuple(t.cuda() for t in batch) if self.args.is_cuda else batch

      with torch.no_grad():
        model_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'demonstration_soft_input_ids': batch[2],
                        'demonstration_soft_label': batch[3],
                        'mlm_label': batch[4],
                        'cls_label': batch[5],
                        'verbalizer': self.verbalizer,
                        'is_eval': True}

        model_output = self.model(**model_inputs)

        final_logits = F.softmax(model_output['logits'], dim=-1)
        final_logits = final_logits.argmax(dim=-1)

      final_logits = final_logits.detach().cpu().tolist() if self.args.is_cuda else final_logits.tolist()
      cls_logits = batch[-1].squeeze(dim=-1)
      cls_logits = cls_logits.detach().cpu().tolist() if self.args.is_cuda else cls_logits.tolist()

      preds.extend(final_logits)
      trues.extend(cls_logits)

    output_dict = compute_metrics_mapping[self.args.task_name](self.args.task_name, np.array(preds), np.array(trues))
    for k, v in output_dict.items():
      logger.info(f'Global Step: {self.global_steps}: {mode}/{k} = {v}')
      self.writer.add_scalar(f'{mode}/{k}', v, self.global_steps)
    self.model.train()

    return output_dict


class DemonstrationPromptInference(BaseTrainer, ABC):
  def __init__(self, args: ArgumentsConfig) -> None:
    super().__init__(args)
    data_processor = processors_mapping[args.task_name]
    test_examples = data_processor.get_test_examples(args.dataset_dir)

    self.base = RobertaForMaskedLM.from_pretrained(args.lm_model)
    self.tokenizer = RobertaTokenizer.from_pretrained(args.lm_model)

    logger.info(f'AutoModelForMaskedLM, AutoTokenizer: {args.lm_model} loaded!')

    soft_tokens = add_special_tokens_and_resize_embedding(self.tokenizer, self.base)

    self.model = ContiDemonPromptModel(self.args, self.base,
                                       total_soft_len=len(list(itertools.chain(
                                         *copy(self.args.demonstration_tokens)))) * self.args.soft_token_length,
                                       verbalizer_idx=self.get_verbalizer(self.args.task_name, self.tokenizer).tolist())

    if args.demonstration_rank_output_path is not None:
      with open(args.demonstration_rank_output_path, 'rb') as f:
        demonstrations = json.load(f)
        mode = 'rank'
    else:
      demonstrations = args.demonstration_tokens
      mode = 'static'

    logger.info(f'demonstration tokens of {self.args.task_name} -> {demonstrations}')

    self.test_dataset, self.test_dataloader = self.get_dataloader(test_examples, self.tokenizer,
                                                                  soft_tokens, demonstrations, mode, False)

    trained_weights = torch.load(os.path.join(self.args.tensorboard_output_dir, 'best_model.pth'), map_location='cpu')
    self.model.load_state_dict(trained_weights)

    if self.args.is_cuda:
      logger.info('Now all parameter is in CUDA option.')
      self.model.cuda()
      self.verbalizer = self.get_verbalizer(self.args.task_name, self.tokenizer).cuda()

  def inference(self):
    self.model.eval()

    preds = []
    trues = []

    hidden_states = []
    cls_hidden_states = []
    labels = []

    for batch in tqdm(self.test_dataloader, desc=f'Test Iteration'):
      batch = tuple(t.cuda() for t in batch) if self.args.is_cuda else batch

      with torch.no_grad():
        model_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'demonstration_soft_input_ids': batch[2],
                        'demonstration_soft_label': batch[3],
                        'mlm_label': batch[4],
                        'cls_label': batch[5],
                        'verbalizer': self.verbalizer,
                        'is_eval': True}

        model_output = self.model(**model_inputs)
        hidden_states.append(model_output['hidden_states'].detach().cpu())
        cls_hidden_states.append(model_output['cls_hidden_states'].detach().cpu())
        labels.append(batch[5].detach().cpu())

        final_logits = F.softmax(model_output['logits'], dim=-1)
        final_logits = final_logits.argmax(dim=-1)

      final_logits = final_logits.detach().cpu().tolist() if self.args.is_cuda else final_logits.tolist()
      cls_logits = batch[-1].squeeze(dim=-1)
      cls_logits = cls_logits.detach().cpu().tolist() if self.args.is_cuda else cls_logits.tolist()

      preds.extend(final_logits)
      trues.extend(cls_logits)

    output_dict = compute_metrics_mapping[self.args.task_name](self.args.task_name, np.array(preds), np.array(trues))
    for k, v in output_dict.items():
      logger.info(f'{k} = {v}')

    return output_dict, hidden_states, cls_hidden_states, labels


if __name__ == '__main__':
  def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


  set_seed(13)

  logger.info(sys.argv[1])
  args = ArgumentsConfig.from_json_file(sys.argv[1])
  logger.info(args.tensorboard_output_dir)

  DemonstrationPromptTrainer(args).train()
