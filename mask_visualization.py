import logging
import os
import random
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn import manifold

from soft_prompt_trainer_wo_demo import SoftPromptTrainerWithoutDemoInference
from src.common.arguments import ArgumentsConfig
from trainer import DemonstrationPromptInference

import seaborn as sns

sns.set_style("darkgrid")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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

  if 'soft-prompting' in sys.argv[1]:
    _, hidden_states, labels = SoftPromptTrainerWithoutDemoInference(args).inference()
  else:
    _, hidden_states, __, labels = DemonstrationPromptInference(args).inference()
  hidden_states = torch.cat(hidden_states, dim=0).numpy()
  labels = torch.cat(labels, dim=0).numpy()

  hidden_states = manifold.TSNE(n_components=2, init='pca', random_state=13).fit_transform(hidden_states)

  fig = plt.figure(figsize=(6, 6))
  # ax1 = fig.add_subplot(2, 1, 1)
  # ax1.set_title('test')
  # ax1.set_xlim(left=-20, right=20)
  # ax1.set_ylim(top=20, bottom=-20)

  hidden_states = torch.from_numpy(hidden_states)
  positive_labels = [idx for idx, _ in enumerate(labels) if _ == 1]
  negative_labels = [idx for idx, _ in enumerate(labels) if _ == 0]
  positive_features = hidden_states[positive_labels].numpy()
  negative_features = hidden_states[negative_labels].numpy()

  plt.scatter(positive_features[:, 0], positive_features[:, 1], c='blue', label='positive', alpha=0.3)
  plt.scatter(negative_features[:, 0], negative_features[:, 1], c='red', label='negative', alpha=0.3)

  plt.legend()
  plt.grid(True)

  logger.info(f'unbiased True:{torch.var(hidden_states, unbiased=True)}')
  logger.info(f'unbiased False:{torch.var(hidden_states, unbiased=False)}')

  if 'soft-prompting' in sys.argv[1]:
    logger.info(os.path.join(args.tensorboard_output_dir, 'soft-prompting-mask-visualization.png'))
    plt.savefig(os.path.join(args.tensorboard_output_dir, 'soft-prompting-mask-visualization.png'))
  else:
    logger.info(os.path.join(args.tensorboard_output_dir, 'demonstration-memory-mask.png'))
    plt.savefig(os.path.join(args.tensorboard_output_dir, 'demonstration-memory-mask.png'))
