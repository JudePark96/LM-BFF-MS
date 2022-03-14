"""https://github.com/ucinlp/autoprompt/blob/master/autoprompt/utils.py"""

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputStorage:
  """
  This object stores the intermediate gradients of the output a the given PyTorch module, which
  otherwise might not be retained.
  """

  def __init__(self, module):
    self._stored_output = None
    module.register_forward_hook(self.hook)

  def hook(self, module, input, output):
    self._stored_output = output

  def get(self):
    return self._stored_output