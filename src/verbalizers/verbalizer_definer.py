import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

Verbalizer = {
  "cola": [' incorrect', 'correct'],
  "mnli": ['No', 'Yes', 'Maybe'],
  "mpqa": [' terrible', ' great'],
  "mrpc": ['No', 'Yes'],
  "mr": [' terrible', ' great'],
  "cr": [' terrible', ' great'],
  "qnli": ['No', 'Yes'],
  "qqp": ['No', 'Yes'],
  "rte": ['Yes', 'No'],
  "snli": ['No', 'Yes', 'Maybe'],
  "sst-2": [' terrible', ' great'],
  "subj": [' subjective', ' objective'],
}
