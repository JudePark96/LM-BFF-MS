import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseConfig:
  def save_config(self, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
      logger.info(f'Experiment config will be saved at {path}')
      json.dump(self.__dict__, f, ensure_ascii=False, indent=2)

  @classmethod
  def from_json_file(cls, path: str):
    with open(path, 'r', encoding='utf-8') as f:
      logger.info(f'Loading Experiment config: {path}')
      json_dict = json.load(f)
    return cls(**json_dict)


@dataclass(frozen=True)
class ArgumentsConfig(BaseConfig):
  train_batch_size: Optional[int] = None
  eval_batch_size: Optional[int] = None
  task_name: Optional[str] = None
  random_seed: Optional[int] = None
  k_shot: Optional[int] = None
  dataset_dir: Optional[str] = None  # We have a DataProcessor!
  learning_rate: Optional[float] = None
  max_grad_norm: Optional[float] = None
  lm_model: Optional[str] = None
  manual_template: Optional[str] = None
  max_seq_length: Optional[int] = None
  gradient_accumulation_steps: Optional[int] = None
  num_train_epochs: Optional[int] = None
  log_step_count_steps: Optional[int] = None
  eval_every_step: Optional[int] = None
  lambda_rate: Optional[float] = None
  demonstration_tokens: Optional[List[List[str]]] = None
  demonstration_rank_output_path: Optional[str] = None
  soft_token_length: Optional[int] = None
  is_cuda: Optional[bool] = None
  warmup_proportion: Optional[float] = None
  tensorboard_output_dir: Optional[str] = None


if __name__ == '__main__':
  ArgumentsConfig(4, 8, 'sst-2', None, None, '../../rsc/eng_original/SST-2/', None, None,
                  'roberta-large', 'It was <mask> .', 256, None, None, None, None,
                  None, None, True, None, None).save_config('../../rsc/experiment_configs/zeroshot/sst-2/32-13-sst2-zeroshot.json')
  pass
  # ).save_config('experiment_configs/SST2/32-13-SST2-HardPrompt-bias-fit-config.json')
  #
  # ArgumentsConfig(
  #   8, 8, 'SST2', 13, 32, './rsc/glue_few_shot/k-shot/SST2/32-13/', 3e-5, 5e-5,
  #   1.0, 'RoBERTa-large', '[T] [T] [T] [T] [T] [X] [M]', 'It was', 'hard-prompt',
  #   'prompting', 256, 1, 50, 10, 20, False, False, False, False, 0.6, './logs/SST2-32-13-NullPrompt-Finetune'
  # ).save_config('experiment_configs/SST2/32-13-SST2-HardPrompt-finetune-config.json')
  #
  # ArgumentsConfig(
  #   8, 8, 'SST2', 13, 32, './rsc/glue_few_shot/k-shot/SST2/32-13/', 3e-5, 5e-5,
  #   1.0, 'RoBERTa-large', '[T] [T] [T] [T] [T] [X] [M]', None, 'cls',
  #   'prompting', 256, 1, 50, 10, 20, False, False, False, False, 0.6, './logs/SST2-32-13-cls-Finetune'
  # ).save_config('experiment_configs/SST2/32-13-SST2-CLS-finetune-config.json')
