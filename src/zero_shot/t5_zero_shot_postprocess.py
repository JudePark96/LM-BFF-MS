import json
import logging
import os
import sys
from collections import defaultdict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
  output_path, task_name = sys.argv[1], sys.argv[2]
  json_list = os.listdir(output_path)

  output_by_rank = []

  for file_path in json_list:
    with open(os.path.join(output_path, file_path), 'rb') as f:
      output_by_rank.append(json.load(f))

  gathered_output_stores = defaultdict()
  final_output_stores = defaultdict()
  original_keys = [k for k, v in output_by_rank[0].items()]

  for k in original_keys:
    gathered_output_stores[k] = {}
    final_output_stores[k] = {}

  for output in output_by_rank:
    for o_k, o_v in output.items():
      for r_o_v in o_v:
        if len(r_o_v[0]) == 1:
          continue
        if r_o_v[0] not in gathered_output_stores[o_k]:
          gathered_output_stores[o_k][r_o_v[0]] = r_o_v[1]
        else:
          gathered_output_stores[o_k][r_o_v[0]] += r_o_v[1]

  # with open(f'./t5_zero_shot_outputs/{task_name}/{task_name}-train-zeroshot-gathered.json', 'w', encoding='utf-8') as f:
  #   gathered_output_stores = sorted(gathered_output_stores.items(), key=lambda x: x[1], reverse=True)[:100]
  #   json.dump(gathered_output_stores, f, ensure_ascii=False, indent=2)
  #
  # exit()

  remove_duplicated_keys = None
  if len(original_keys) == 2:
    remove_duplicated_keys = list(set(gathered_output_stores['0']) & set(gathered_output_stores['1']))
  elif len(original_keys) == 3:
    a_b = set(gathered_output_stores['0']) & set(gathered_output_stores['1'])
    a_c = set(gathered_output_stores['0']) & set(gathered_output_stores['2'])
    b_c = set(gathered_output_stores['1']) & set(gathered_output_stores['2'])

    remove_duplicated_keys = a_b | a_c | b_c

  remove_duplicated_keys = list(remove_duplicated_keys)

  for key, value in gathered_output_stores.items():
    for inner_key, inner_value in value.items():
      if '<unk>' in inner_key:
        continue

      if inner_key not in remove_duplicated_keys:
        final_output_stores[key][inner_key] = inner_value

  for k, v in final_output_stores.items():
    final_output_stores[k] = sorted(final_output_stores[k].items(), key=lambda x: x[1], reverse=True)

  with open(f'./t5_zero_shot_outputs/{task_name}/{task_name}-train-zeroshot-gathered.json', 'w', encoding='utf-8') as f:
    json.dump(final_output_stores, f, ensure_ascii=False, indent=2)