import logging

from transformers import PreTrainedTokenizer, PreTrainedModel, RobertaForMaskedLM, RobertaTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def add_special_tokens_and_resize_embedding(tokenizer: PreTrainedTokenizer,
                                            model: PreTrainedModel) -> str:
  logger.info(f'Soft Tokens: [T]')
  logger.info(f'before adding special tokens, vocab size is as follows: {len(tokenizer)}')
  tokenizer.add_special_tokens({
    'additional_special_tokens': ['[T]']
  })
  logger.info(f'after adding special tokens, vocab size is as follows: {len(tokenizer)}')
  model.resize_token_embeddings(len(tokenizer))

  assert len(tokenizer) == model.get_input_embeddings().weight.data.shape[0]

  logger.info('adding special tokens and resizing embedding has done!')

  return '[T]'


if __name__ == '__main__':
  model = RobertaForMaskedLM.from_pretrained('roberta-large')
  tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

  add_special_tokens_and_resize_embedding(tokenizer, model)

  print(tokenizer.special_tokens_map)