import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_str_to_discrete_label(task: str, label_str: str) -> int:
  if task == 'mnli' or task == 'snli':
    if label_str == 'contradiction':
      label = 0
    elif label_str == 'entailment':
      label = 1
    elif label_str == 'neutral':
      label = 2
    else:
      raise ValueError(f'Wrong label: {label_str}')
  elif task == 'qnli' or task == 'rte':
    if label_str == 'entailment':
      label = 0
    elif label_str == 'not_entailment':
      label = 1
    else:
      raise ValueError(f'Wrong label: {label_str}')
  else:
    label = int(label_str)

  return label
