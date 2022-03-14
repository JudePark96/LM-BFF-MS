import json
import logging
import os
import sys
import torch
import torch.nn.functional as F

from collections import defaultdict, Counter
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForMaskedLM

sys.path.append(os.getcwd() + "/../../")  # noqa: E402

from src.common.arguments import ArgumentsConfig
from src.data_utils.processors import processors_mapping, num_labels_mapping
from src.feature_utils.convert_examples_to_normal_prompt_features import convert_examples_to_prompt_features

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
  args = ArgumentsConfig.from_json_file(sys.argv[1])
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

  data_processor = processors_mapping[args.task_name]
  train_examples = data_processor.get_train_examples(args.dataset_dir)

  tokenizer = RobertaTokenizer.from_pretrained(args.lm_model)
  model = RobertaForMaskedLM.from_pretrained(args.lm_model)
  features = convert_examples_to_prompt_features(train_examples,
                                                 tokenizer,
                                                 args.task_name,
                                                 args.manual_template,
                                                 500 // 2,
                                                 model.config.max_position_embeddings)[:1000]
                                                 # 512)
  dataset = TensorDataset(
    torch.tensor([f.input_ids for f in features], dtype=torch.long),
    torch.tensor([f.attention_mask for f in features], dtype=torch.float),
    torch.tensor([f.mlm_label for f in features], dtype=torch.long),
    torch.tensor([f.cls_label for f in features], dtype=torch.long)
  )

  labels = [i for i in range(num_labels_mapping[args.task_name])]
  stop_words = ['Ġa', 'the', 'Ġis', 'is', 'Ġare', 'are', 'they', 'Ġthey', 'you', 'Ġyou', 'Ġmy', 'my', 'Ġour', 'our',
                'Ġmyself', 'Ġourselves', 'Ġin', 'in', 'Ġon', 'on', 'Ġalso' 'also', 'Ġan', 'Ġmy', 'Ġthe', 'Ġis', 'Ġare',
                'Ġthey', 'Ġyou', 'Ġour', 'Ġ<', 'Ġ>', ':', 'Ġ:', '?', 'Ġ?', 'Ġhim', 'him', 'her', 'Ġher', 'Ġhis', 'his ',
                'he', 'Ġhe', 'she', 'Ġshe', 'them', 'Ġthem', 'from', 'Ġfrom']

  stop_words.extend([t[1] for t in tokenizer.special_tokens_map.items()])

  dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

  if args.is_cuda:
    model.cuda()
    model.eval()

  top_k = 40
  outputs = defaultdict()
  prob_outputs = defaultdict()
  final_outputs = defaultdict()

  for label in labels:
    outputs[label] = {}
    prob_outputs[label] = {}
    final_outputs[label] = {}

  for batch in tqdm(dataloader, total=len(dataloader)):
    batch = tuple(t.cuda() for t in batch)

    with torch.no_grad():
      model_inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
      model_output = model(**model_inputs)
      final_logits = F.softmax(model_output.logits[batch[2] > 0], dim=-1)

    topk_prob_value, topk_prob_idx = torch.topk(final_logits, k=top_k)
    cls_label = batch[-1].detach().cpu().tolist()
    for i, (value, top, cls) in enumerate(zip(topk_prob_value, topk_prob_idx, cls_label)):
      decoded = [tokenizer.convert_ids_to_tokens(j) for j in top.detach().cpu().tolist()]
      for idx, d in enumerate(decoded):
        # if d in [str(i) for i in range(10)]:
        #   continue
        if '0' in d or '1' in d or '2' in d or '3' in d or '4' in d or '5' in d or '6' in d or '7' in d or '8' in d or \
          '9' in d:
          continue
        if 'and' in d or 'or' in d or 'And' in d or 'Or' in d or 'OR' in d:
          continue
        if '.' in d or '>' in d or '<' in d or ':' in d or '?' in d or '\"' in d or '-' in d or ',' in d or '=' in d or \
          d in stop_words:
          continue

        if d not in outputs:
          outputs[cls][d] = 1
          prob_outputs[cls][d] = value[idx].detach().cpu().item()
        else:
          outputs[cls][d] += 1
          prob_outputs[cls][d] += value[idx].detach().cpu().item()

  common_k = 40

  for k, v in outputs.items():
    outputs[k] = Counter(v).most_common(common_k)

  for k, v in outputs.items():
    final_outputs[k] = {i[0]: prob_outputs[k][i[0]] / i[1] for i in v}
    # final_outputs[k] = {i[0]: i[1] for i in v}
    final_outputs[k] = sorted(final_outputs[k].items(), key=lambda x: x[1], reverse=True)

  with open(f'./zero_shot_outputs/{args.task_name}-train-zeroshot-test.json', 'w', encoding='utf-8') as f:
    json.dump(final_outputs, f, ensure_ascii=False, indent=2)
