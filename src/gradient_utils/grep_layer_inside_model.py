import logging

from transformers import RobertaForMaskedLM

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_hidden_states(model):
  if isinstance(model, RobertaForMaskedLM):
    return model.roberta.encoder.layer[-1].output.LayerNorm
  else:
    raise NotImplementedError(f'{model} not currently supported')


def get_mlm_heads(model):
  if isinstance(model, RobertaForMaskedLM):
    return model.lm_head
  else:
    raise NotImplementedError(f'{model} not currently supported')


def get_mlm_decoder(model):
  if isinstance(model, RobertaForMaskedLM):
    return model.lm_head.decoder
  else:
    raise NotImplementedError(f'{model} not currently supported')


def get_final_embeddings(model):
  if isinstance(model, RobertaForMaskedLM):
    return model.lm_head.layer_norm
  else:
    raise NotImplementedError(f'{model} not currently supported')
