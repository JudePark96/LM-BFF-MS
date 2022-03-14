import logging
import sys
import numpy as np

from tqdm import tqdm
from transformers import RobertaTokenizer
from src.data_utils.processors import processors_mapping

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
  task, lm, dataset_dir = sys.argv[1], sys.argv[2], sys.argv[3]
  tokenizer = RobertaTokenizer.from_pretrained(lm)
  data_processor = processors_mapping[task]
  train_examples = data_processor.get_train_examples(dataset_dir)

  seq_len_stores = []

  for idx, example in enumerate(tqdm(train_examples)):
    # print(idx, example.text_a)
    if example.text_a == 'nan':
      text_a_len = len(tokenizer.tokenize(' '))
    else:
      text_a_len = len(tokenizer.tokenize(example.text_a))

    text_b_len = 0

    if example.text_b is not None:
      text_b_len = len(tokenizer.tokenize(example.text_b))

    seq_len_stores.append(text_a_len + text_b_len)

  print(f'Mean Length: {np.mean(seq_len_stores)}')
  print(f'Max Length: {np.max(seq_len_stores)}')
  print(f'Min Length: {np.min(seq_len_stores)}')

  """
  MNLI
  Mean Length: 36.67046768287404
  Max Length: 421
  Min Length: 3
  
  SNLI
  Mean Length: 23.270471287864034
  Max Length: 122
  Min Length: 5
  """



