# LM-BFF-MS
Official Implementation for ACL 2022 Paper "LM-BFF-MS: Improving Few-Shot Fine-tuning of Language Models based on Multiple Soft Demonstration Memory"

This code is reimplemented as a fork of [LM-BFF](https://github.com/princeton-nlp/LM-BFF).

# Requirements

To run our code, please install all the dependency packages by using the following command:

```shell script
pip3 install -r requirements.txt
```

Note that we support only **PyTorch**.

# Running

```shell script
python3 trainer.py configs/sst2/16-13-sst2-conti-demon-prompting.json
```

# Contact

Do not hesitate to ask a question. Send me your question below email!
```
Eunhwan Park (judepark@jbnu.ac.kr)
```

# Citation

```
@inproceedings{park-etal-2022-lm,
    title = "{LM}-{BFF}-{MS}: Improving Few-Shot Fine-tuning of Language Models based on Multiple Soft Demonstration Memory",
    author = "Park, Eunhwan  and
      Jeon, Donghyeon  and
      Kim, Seonhoon  and
      Kang, Inho  and
      Na, Seung-Hoon",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.34",
    pages = "310--317",
    abstract = "LM-BFF (CITATION) achieves significant few-shot performance by using auto-generated prompts and adding demonstrations similar to an input example. To improve the approach of LM-BFF, this paper proposes \textbf{LM-BFF-MS}{---}\textbf{b}etter \textbf{f}ew-shot \textbf{f}ine-tuning of \textbf{l}anguage \textbf{m}odels with \textbf{m}ultiple \textbf{s}oft demonstrations by making its further extensions, which include 1) prompts with \textit{multiple demonstrations} based on automatic generation of multiple label words; and 2) \textit{soft demonstration memory} which consists of multiple sequences of \textit{globally shared} word embeddings for a similar context. Experiments conducted on eight NLP tasks show that LM-BFF-MS leads to improvements over LM-BFF on five tasks, particularly achieving 94.0 and 90.4 on SST-2 and MRPC, respectively.",
}
```
