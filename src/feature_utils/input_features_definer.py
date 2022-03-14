import logging
from dataclasses import dataclass
from typing import List, Optional

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BasePromptInputFeatures:
  input_ids: List[int]
  attention_mask: Optional[List[int]] = None
  token_type_ids: Optional[List[int]] = None
  mlm_label: Optional[List[int]] = None
  cls_label: Optional[List[int]] = None


@dataclass(frozen=True)
class PromptWithContinuousDemonstrationFeatures:
  input_ids: List[int]
  attention_mask: Optional[List[int]] = None
  token_type_ids: Optional[List[int]] = None
  demonstration_soft_input_ids: Optional[List[int]] = None
  demonstration_soft_label: Optional[List[int]] = None
  mlm_label: Optional[List[int]] = None
  cls_label: Optional[List[int]] = None
