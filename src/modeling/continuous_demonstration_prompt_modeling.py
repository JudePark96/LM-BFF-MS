import logging

import torch
from transformers import PreTrainedModel

from src.common.arguments import ArgumentsConfig
from src.gradient_utils.grep_layer_inside_model import get_hidden_states, get_mlm_heads, get_mlm_decoder
from src.gradient_utils.hook_storage import OutputStorage

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import torch.nn as nn


class ContiDemonPromptModel(nn.Module):
  def __init__(self,
               args: ArgumentsConfig,
               base_model: PreTrainedModel,
               total_soft_len: int,
               verbalizer_idx: int) -> None:
    super().__init__()
    self.args = args
    self.total_soft_len = total_soft_len
    self.base_model = base_model
    self.conti_demon_embed = nn.Embedding(1 + self.total_soft_len, self.base_model.config.hidden_size, padding_idx=0)
    self.ce_loss = nn.CrossEntropyLoss()
    self.mlm_head = get_mlm_heads(self.base_model)
    self.hidden = OutputStorage(get_hidden_states(self.base_model))

    verbalizer_weights = get_mlm_decoder(self.base_model).weight.data[verbalizer_idx, :].clone()
    logger.info(f'verbalizer length => {len(verbalizer_idx)}')

    self.cls_layer = nn.Linear(self.base_model.config.hidden_size, len(verbalizer_idx))
    self.cls_layer.weight.data = verbalizer_weights

  def forward(self,
              input_ids: torch.Tensor,
              attention_mask: torch.Tensor,
              demonstration_soft_input_ids: torch.Tensor,
              demonstration_soft_label: torch.Tensor,
              mlm_label: torch.Tensor,
              cls_label: torch.Tensor,
              verbalizer: torch.Tensor,
              is_eval: bool = False):
    bs, seq_len = input_ids.size()
    inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
    demon_embeds = self.conti_demon_embed(demonstration_soft_input_ids)
    inputs_embeds[demonstration_soft_label > 0] = demon_embeds.view(-1, self.base_model.config.hidden_size)
    self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    hidden_states = self.hidden.get()
    logits = self.mlm_head(hidden_states)[mlm_label > 0][:, verbalizer]

    if self.args.lambda_rate != 0.0:
      cls = self.cls_layer(hidden_states[:, 0, :])

    if is_eval:

      return {
        'logits': logits,
        'hidden_states': hidden_states[mlm_label > 0],  # Note that this is for analysis
        'cls_hidden_states': hidden_states[:, 0, :]  # Note that this is for analysis
      }
    else:
      if self.args.lambda_rate != 0.0:
        return {
          'loss': self.ce_loss(logits, cls_label) + (self.args.lambda_rate * self.ce_loss(cls, cls_label))
        }
      else:
        return {
          'loss': self.ce_loss(logits, cls_label)
        }
