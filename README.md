# NewsLiveblog‑TLS 🗞️⏱️

*A Time‑Sensitive Benchmark for Extractive Timeline Summarization*

---

## 1  Project at a Glance

| Item                | Value                                                                     |
| ------------------- | ------------------------------------------------------------------------- |
| **Language**        |  English                                                                  |
| **Domain**          |  Breaking / World News (*The Guardian* liveblogs)                         |
| **Primary Task**    |  Extractive Timeline Summarization (TLS)                                  |
| **Secondary Tasks** |  Key‑event detection · Temporal sentence classification                   |

---

## 2  Data Format & Folder Layout

```
NewsLiveblog-TLS/
├── data/
│   ├── world_news.json     # 1 473 liveblogs (≈ 124 k sentences)
│   ├── data_collection.ipynb  # HTML scraping via TheGuardian API
│   └── preprocess.py          # sentence‑level oracle labelling (TAEGS)
└── README.md
```

Each line in `*.jsonl` is a **single liveblog** (UTF‑8, no BOM):

```json
{
  "document":   ["sentence_1", …, "sentence_N"],
  "timeline":   ["YYYY‑MM‑DDThh:mm:ssZ", …] ,   # length = N
  "summary":    ["editor bullet‑1", …],          # key‑point list
  "key_timeline": ["ISO‑8601‑1", …]              # oracle dates
}
```

---

## 3  Construction Pipeline

| Stage                    | Tool / Script            | Key Decisions                                                                                                                                      |         |   |         |               |
| ------------------------ | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | - | ------- | ------------- |
| **1  Retrieval**         |  `data_collection.ipynb` | Queried *The Guardian* Content API (world news, type = "liveblog"), fetched latest 2 000 posts (2012 → 2024).                                      |         |   |         |               |
| **2  HTML Parse**        |  BeautifulSoup           | Kept `<div class="block-elements">`; removed nav, ads, footers.                                                                                    |         |   |         |               |
| **3  Noise Reduction**   | regex rules              | Dropped multi‑topic “live updates”, sports blogs, in‑progress threads.                                                                             |         |   |         |               |
| **4  Content Trim**      | heuristics               | Removed titles, image captions, Twitter embeds, “Email link” strings; kept finished blogs only.                                                    |         |   |         |               |
| **5  Advanced Cleaning** | word‑ratio filter        | Deleted rows where **compression ratio** (                                                                                                         | summary | / | article | words) > 0.4. |
| **6  Oracle Labelling**  |  `preprocess.py` (TAEGS) | spaCy sentence split + NER (`DATE`) → greedy maximise **ROUGE‑L** under date‑overlap constraint; outputs `extract_label` indices + `key_timeline`. |         |   |         |               |

> **TAEGS** (Temporal‑Alignment Enhanced Greedy Selection) is fully described in report §4.3; pseudocode in *preprocess.py* matches Algorithm 1. The final corpus contains **1 473** cleaned liveblogs.

---

## 4  Corpus Statistics

| Metric                         | Value                               |
| ------------------------------ | ----------------------------------- |
| Documents                      | **1 473**                           |
| **Avg tokens / doc**           |  2 281.21                           |
| **Avg sentences / doc**        |  89.50                              |
| **Avg dates / doc**            |  13.27                              |
| **Avg sentences / date**       |  6.74                               |
| **Avg summary tokens**         |  151.40                             |
| **Avg summary sentences**      |  15.04                              |
| **Sentence‑level compression** |  × 9.16                             |
| **Date‑level compression**     |  × 4.08                             |
| Vocabulary (doc / summary)     |  79 866 / 16 140 unique tokens      |
| Time span per liveblog         |  mean ≈ 4 h 41 m (max < 12 h)       |
| Coverage years                 |  2012 – 2024 (peaks in 2014 & 2022) |

---

## 5  Suggested Tasks & Metrics

| Task                       | Input                   | Output                       | Recommended Metrics               |
| -------------------------- | ----------------------- | ---------------------------- | --------------------------------- |
| **Timeline Summarization** | `document` + `timeline` | pick *k* sentences per date  | ROUGE‑1/2/L · **Date‑F1** · AR1‑F |
| **Key‑Event Detection**    | same                    | predict `key_timeline` dates | Date‑F1, Mean Delay               |
| **Temporal Sentence Cls.** | single sentence         | binary label                 | F1‑macro                          |

---

## 6  Quick Start (PyTorch)

```python
import json, pathlib
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased")

with pathlib.Path("world_news.json").open() as f:
    sample = json.loads(f.readline())

print("Sentences:", len(sample["document"]))
print("First sentence:", sample["document"][0])
print("First timestamp:", sample["timeline"][0])

enc = tok(sample["document"][:16],
          padding=True, truncation=True,
          max_length=32, return_tensors="pt")
out = bert(**enc)
print(out.last_hidden_state.shape)      # (batch, seq, 768)
```

---

## 7  License

This dataset redistributes text snippets from **The Guardian** under UK *fair dealing* / US *fair use* for **non‑commercial research** only.
For commercial usage please obtain permission from the original publisher.



### Acknowledgements

Data collection scripts build upon *The Guardian* Open API. Sentence splitting & NER use **spaCy 3.7** (`en_core_web_sm`). Temporal alignment oracle is implemented in *preprocess.py* (see §4.3 of the accompanying report).
