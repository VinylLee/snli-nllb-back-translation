# SNLI Back Translation 数据质量改造任务书

> 交付对象：GPT-5.6 Luna（代码实现版）  
> 目标仓库：https://github.com/VinylLee/snli-nllb-back-translation  
> 基线检查日期：2026-09-07

## 任务一句话

不要只给优化建议。请直接修改 `snli-nllb-back-translation` 仓库，把当前“生成后直接继承 label”的 back-translation 流程改造成：

```text
生成候选
→ 语义等价检查
→ NLI label 保持检查
→ 风险规则过滤
→ accepted / rejected 分流保存
```

核心优先级：**先提高 label-preserving precision，再提高 augmentation diversity。**

---

## 给 Luna 的执行要求

1. 先阅读当前脚本、测试和 18 条质量样例，确认现状与本文一致。
2. 直接修改代码并新增必要模块/测试；不要只返回伪代码或建议。
3. 默认保持向后兼容：原有 CLI 能继续工作，但推荐的新研究流程必须启用质量过滤。
4. 实现后运行单元测试，并用 `--max-samples 18` 做一次小规模质量回归。
5. 最终报告必须列出：
   - changed files；
   - 关键设计与兼容性说明；
   - 测试命令与测试结果；
   - 18 条样例的 accepted/rejected 原因摘要；
   - 仍存在的限制和下一步建议。

---

# 1. 当前仓库现状与问题定义

当前仓库使用 `facebook/nllb-200-distilled-600M`，将英文分别翻译到 pivot language，再翻回英文。

README 描述目标是“preserving the label”，但当前代码实际只是复制原 label，并没有验证翻译后的 `premise-hypothesis` 关系是否仍属于该 label。

| 位置 / 主题 | 当前行为 | 质量风险 |
|---|---|---|
| `scripts/back_translate_nllb.py` | 默认独立 back-translate premise 与 hypothesis，随后直接复制 label | 两侧同时可能发生语义漂移，但 label 没有重新验证 |
| generation | `num_beams=4, do_sample=False` | 生成确定性较强；seed 对当前 beam search 几乎没有实际多样性作用 |
| 输入处理 | `truncation=True` | 超长输入会静默截断，可能改变语义且没有留下质量标记 |
| 内存 / 恢复 | `records = list(read_records(...))` | 完整训练集加载进内存；输出最后统一写入，checkpoint/resume 能力有限 |
| `tests/test_back_translate_nllb.py` | 主要验证读取、schema 和 label 字段复制 | 没有验证 semantic / label preservation |
| `snli_quality_18.jsonl` | 共 18 条；若前 3 条是 smoke，则 `source_index=3..17` 是 15 条追加质量样例 | 已经能观察到真实语义漂移和 label 破坏 |

## 1.1 已观察到的失败模式

现有增强样例已经说明：**单句看起来像正常英语，并不代表它仍是有效 NLI 数据。**

### `source_index=4`

原 label 是 entailment，但增强后的 hypothesis 出现：

```text
There's kids in the house.
```

premise 只描述孩子在镜头前活动，并不能推出“在屋里”。因此原 entailment 已不可靠，可能变成 neutral。

### `source_index=5`

原 label 是 contradiction。增强后出现类似：

```text
smiling children
vs
rubbing their eyebrows
```

这两件事可以同时成立，因此原 contradiction 被削弱甚至消失。

### `source_index=9`

出现：

```text
daughter retires from work
```

原义发生明显漂移。即使 pair-level label 碰巧没变，也不应把它视为原句的等义 paraphrase。

### `source_index=10`

出现：

```text
turning on a hamburger
```

这属于明显的语义/流畅度异常，应被 semantic-equivalence 或 corruption 规则拒绝。

## 核心诊断

当前 pipeline 隐含假设：

```text
back translation
≈ semantic equivalence
≈ NLI label preservation
```

15 条追加样例已经证明这个假设不成立。

真正应该优化的是：**candidate acceptance / filtering**，而不只是翻译模型本身。

---

# 2. 改造目标与非目标

## 2.1 必须达到的目标

- 每条增强数据在写入 accepted 输出前，都必须经过可解释的质量判定。
- 默认降低双侧同时漂移风险：支持只改 premise、只改 hypothesis，以及分别生成两个单侧候选。
- 检查原句与 back-translated 句子的语义等价性，而不是只看字符相似度。
- 检查增强后的 `premise-hypothesis` 关系是否仍与 SNLI gold label 一致，并设置置信度阈值。
- 对 negation、数字、量词、modal、时间/空间方向词等 NLI 高风险 cue 做显式保护。
- accepted 与 rejected 都要保留 provenance、分数和拒绝原因，便于复现实验和人工抽检。
- 真正以 batch streaming 方式边生成、边过滤、边写盘，支持中断恢复。
- 新增单元测试验证 semantic / label filtering 行为，而不是只验证 label 字段被复制。

## 2.2 本轮不要优先做的事情

- 不要先换更大的 NLLB 模型来掩盖 pipeline 问题。
- 不要一开始就启用高 temperature sampling；先保证 fidelity，再增加 diversity。
- 不要把 BLEU / cosine similarity 当作最终质量标准。
- 不要修改原始 SNLI train / validation / test 文件；增强输出必须单独保存。
- 不要使用 validation/test 的 gold label 或文本去训练过滤器。

---

# 3. 推荐的新流水线

```text
original record
  |
  +-- validate schema / label / length
  |
  +-- generate candidate(s) with NLLB
  |     +-- BT(premise), original hypothesis
  |     +-- original premise, BT(hypothesis)
  |
  +-- semantic-equivalence check
  |     original sentence <-> BT sentence
  |
  +-- logical-cue guard
  |
  +-- NLI relation-preservation check
  |     on augmented premise-hypothesis pair
  |
  +-- diversity / trivial-copy check
  |
  +-- ACCEPT -> accepted.jsonl
  |
  +-- REJECT -> rejected.jsonl + reasons
```

## 3.1 默认采用 asymmetric / separate augmentation

不要默认只生成：

```text
BT(P), BT(H), label
```

优先从一条原始样本生成两个单侧候选：

```text
candidate A: (BT(P), H, label)
candidate B: (P, BT(H), label)
```

好处：

1. 每个候选只有一个句子可能发生漂移；
2. NLI 关系更容易验证；
3. 一条原始 SNLI 可以产生两个不同候选；
4. 更容易定位是 premise 还是 hypothesis 的翻译导致 label drift。

保留兼容模式：

```text
--augmentation-mode both
```

但不要把 `both` 作为高质量增强的推荐默认模式。

---

# 4. 必须实现的代码改动（MVP）

## 4.1 拆分模块

不要继续把所有逻辑堆在单个脚本中。

建议最少形成：

```text
scripts/back_translate_nllb.py   # CLI / orchestration / streaming
scripts/quality_filter.py        # verifier + cue guard + decision

tests/test_back_translate_nllb.py
tests/test_quality_filter.py
```

文件名可调整，但职责必须清晰。

---

## 4.2 输入校验与静默截断处理

必须实现：

- 只接受 `label in {0, 1, 2}`。
- 遇到 `-1` 或其他非法 label 时默认 skip/reject，并记录：

```text
invalid_label
```

- `premise` / `hypothesis` 必须是非空字符串。
- 不允许继续使用类似 `str(None)` 的静默转换。
- tokenize 前先检查长度。
- 如果超过 `--max-input-tokens`，默认拒绝并记录：

```text
input_too_long
```

- 只有显式传入：

```text
--allow-truncation
```

才允许截断。

如果发生截断，metadata 必须记录：

```json
{
  "was_truncated": true
}
```

---

# 5. 语义等价检查：双向 entailment

推荐直接使用 `transformers`：

```python
AutoTokenizer
AutoModelForSequenceClassification
```

不要为了这一版 MVP 额外引入 `sentence-transformers`。

默认 verifier 建议：

```text
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

该模型基于 MultiNLI / FEVER-NLI / ANLI 等数据训练，不直接依赖 SNLI，可作为相对独立的初始过滤器。

## 5.1 关键要求

**不要硬编码 verifier logits 的 label 顺序。**

必须根据：

```python
model.config.id2label
model.config.label2id
```

解析 entailment / neutral / contradiction。

## 5.2 双向语义检查

实现类似：

```python
def semantic_equivalent(original: str, candidate: str) -> Decision:
    p_forward = P(entailment | original, candidate)
    p_backward = P(entailment | candidate, original)

    accept = (
        p_forward >= semantic_threshold
        and p_backward >= semantic_threshold
    )

    return decision_with_scores_and_reason
```

也就是同时要求：

```text
original -> candidate = entailment
candidate -> original = entailment
```

这样可以降低以下问题被误接受的概率：

```text
hugging -> kissing
work ends -> retires from work
outside -> inside
```

## 5.3 默认阈值

先使用：

```text
semantic_threshold = 0.80
```

但必须做成 CLI 参数。

**0.80 只是工程起始值，不是最终最优阈值。**

后续需要用人工审核集校准。

---

# 6. NLI label preservation 检查

语义等价检查通过后，再检查增强后的最终 pair：

```text
augmented_premise
augmented_hypothesis
```

是否仍然符合原 gold label。

SNLI label 映射：

```text
0 = entailment
1 = neutral
2 = contradiction
```

但 verifier 内部 logits 顺序仍然必须从 config 读取。

实现逻辑：

```python
probs = verifier.predict(augmented_premise, augmented_hypothesis)
pred_label = argmax(probs)
gold_prob = probs[gold_label]

accept = (
    pred_label == gold_label
    and gold_prob >= nli_threshold
)
```

默认：

```text
nli_threshold = 0.80
```

同样必须可配置。

## 拒绝原因

如果 predicted label 和 gold label 不一致：

```text
label_flip
```

如果 label 一致，但 gold probability 低于阈值：

```text
low_nli_confidence
```

---

# 7. NLI 专用 logical-cue guard

增加一个轻量、可解释、可测试的规则层。

至少检查下面这些 cue：

| 类别 | 示例 | 风险 |
|---|---|---|
| 否定 | `not`, `no`, `never`, `nobody`, `nothing`, `without`, `n't` | 否定新增/丢失可能直接翻转 entailment / contradiction |
| 数字 | `1/2/3`, `one/two/three`, `couple` | 数量变化对 NLI 很敏感 |
| 量词 | `all`, `some`, `any`, `every`, `few`, `several`, `many`, `most` | 范围改变可能破坏推理关系 |
| 模态 | `may`, `might`, `can`, `could`, `must`, `should`, `will` | 可能性 / 必然性发生变化 |
| 时间 | `before`, `after`, `during`, `first`, `later`, `already` | 事件顺序可能改变 |
| 空间 / 方向 | `in`, `inside`, `outside`, `on`, `under`, `over`, `behind`, `in front of` | 位置关系非常容易造成 label drift |

MVP 先使用：

```text
regex / token matching
```

即可。

不要为了这一层引入大型 NER 或复杂 NLP 依赖。

如果原句与候选在高风险 cue 上发生明显新增、删除或替换，默认拒绝或至少产生 hard flag：

```text
logical_cue_changed
```

---

# 8. 低成本 fluency / corruption 规则

至少拒绝：

- 空字符串；
- 纯标点；
- 明显重复片段；
- 异常字符比例过高；
- tokenizer / generation 得到明显无效输出。

同时记录长度变化，例如：

```text
length_ratio = len(candidate) / len(original)
```

可以先把以下范围标记为风险：

```text
length_ratio < 0.5
length_ratio > 2.0
```

建议先作为 flag，而不是绝对 hard reject。

不要尝试用复杂 grammar checker 替代 semantic verifier。

例如：

```text
turning on a hamburger
```

这种错误应该主要由 semantic-equivalence / corruption 检查捕获。

---

# 9. Diversity：避免“高质量但完全没变化”

augmentation 不能只是复制原句。

增加简单的 normalized edit / change ratio。

例如新增参数：

```text
--min-change-ratio
```

起始值可以设为：

```text
0.03 ~ 0.05
```

如果候选与原句近乎完全相同，则标记：

```text
trivial_copy
```

必须允许关闭该规则。

注意：**change ratio 只用于 diversity，不是语义正确性的指标。**

---

# 10. 输出数据与可追溯性

建议同时输出：

```text
accepted.jsonl
rejected.jsonl
```

CLI：

```text
--accepted-output
--rejected-output
```

为了兼容旧 CLI，如果只给：

```text
--output
```

则：

- `--output` 代表 accepted；
- rejected 自动写到同目录，例如：

```text
xxx.rejected.jsonl
```

## 10.1 推荐输出 schema

```json
{
  "source_index": 123,
  "label": 0,

  "original_premise": "...",
  "original_hypothesis": "...",

  "premise": "...",
  "hypothesis": "...",

  "augmentation": "back_translation",
  "augmented_field": "premise",
  "pivot_lang": "fra_Latn",

  "quality": {
    "semantic_forward_entailment": 0.94,
    "semantic_backward_entailment": 0.92,
    "nli_predicted_label": 0,
    "nli_gold_probability": 0.96,
    "logical_cue_changes": [],
    "change_ratio": 0.21,
    "accepted": true,
    "reasons": []
  },

  "provenance": {
    "translation_model": "facebook/nllb-200-distilled-600M",
    "translation_model_revision": "...",
    "verifier_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "verifier_model_revision": "...",
    "generation": {
      "num_beams": 4,
      "do_sample": false
    }
  }
}
```

如果使用：

```text
--no-metadata
```

可以保留旧 schema 的兼容行为。

但推荐的高质量研究流程**不要使用 `--no-metadata`**，否则会丢失审计信息。

---

# 11. CLI 设计要求

在不破坏现有参数的前提下，增加：

| 参数 | 行为 |
|---|---|
| `--augmentation-mode` | `separate / premise / hypothesis / both`；推荐默认 `separate` |
| `--quality-filter` | `on/off`；推荐研究流程默认 `on` |
| `--verifier-model` | 默认 `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| `--semantic-threshold` | 默认 `0.80` |
| `--nli-threshold` | 默认 `0.80` |
| `--min-change-ratio` | 默认 `0.03` 或 `0.05` |
| `--allow-truncation` | 默认 `false` |
| `--rejected-output` | 可选，默认从 accepted output 自动派生 |
| `--filter-batch-size` | NLI verifier batch size |

如果因为严格向后兼容暂时不能把 `--quality-filter on` 设为代码默认值，则 README 中必须把启用质量过滤的命令作为**推荐默认用法**。

## 11.1 推荐命令

```bash
python scripts/back_translate_nllb.py \
  --input data/nli/original_dataset/snli/train.json \
  --output data/nli/back_translated/snli_train_fra.accepted.jsonl \
  --rejected-output data/nli/back_translated/snli_train_fra.rejected.jsonl \
  --pivot-lang fra_Latn \
  --augmentation-mode separate \
  --quality-filter on \
  --semantic-threshold 0.80 \
  --nli-threshold 0.80 \
  --batch-size 16 \
  --filter-batch-size 32 \
  --device cuda:0 \
  --dtype float16
```

---

# 12. Streaming、checkpoint 与 reproducibility

必须改掉：

```python
records = list(read_records(...))
```

改成固定 chunk 流式处理，例如：

```text
读取 64 / 128 条
→ 翻译
→ 过滤
→ 写 accepted / rejected
→ flush
→ 下一批
```

要求：

- 每个 chunk 完成后 flush accepted/rejected；
- 中途崩溃时最多损失一个 chunk；
- `--resume` 不能只根据 accepted 行数判断进度；
- rejected candidate 同样属于“已处理”。

## 12.1 candidate ID

为每个 candidate 生成稳定 ID，例如：

```python
candidate_id = f"{source_index}:{augmented_field}:{pivot_lang}"
```

resume 应基于：

```text
source_index + augmented_field + pivot_lang
```

或等价稳定 candidate ID。

## 12.2 Reproducibility metadata

必须记录：

- translation model name；
- translation model revision；
- verifier model name；
- verifier model revision；
- generation config；
- pivot language；
- thresholds；
- augmented field。

如果 revision 无法获取，也显式写：

```json
null
```

而不是省略字段。

另外：当前 `do_sample=False` 时，seed 不负责 beam search 的多样性。README 不要暗示 seed 能控制 deterministic beam 输出的多样性。

---

# 13. 测试要求：必须新增“语义行为测试”

单元测试**不要联网下载大模型**。

使用 fake/stub verifier 返回固定概率，使测试：

- 快速；
- 稳定；
- 可重复；
- CI 可执行。

至少覆盖：

| 测试主题 | 验收行为 |
|---|---|
| schema | JSON / JSONL 读取保持原有行为 |
| label validation | `label=-1` 或非法值被 skip/reject，并有 `invalid_label` |
| separate mode | 一条原始记录生成 premise-only 和 hypothesis-only 两个 candidate；同 source_index，不同 candidate_id |
| semantic drift | 双向 entailment 任一方向低于阈值 -> `semantic_drift` |
| label flip | verifier 预测 label 与 gold 不同 -> `label_flip` |
| low confidence | label 正确但 gold probability 低 -> `low_nli_confidence` |
| logical cue | not/never、数字、量词等高风险 cue 新增/丢失 -> `logical_cue_changed` |
| truncation | 超 token 长度且未传 `--allow-truncation` -> `input_too_long` |
| resume | accepted + rejected 都计入已处理，恢复后不重复生成 |
| metadata | accepted/rejected 都包含 scores、reasons、model/pivot/source_index/augmented_field |

---

# 14. 18 条样例的回归验收

## 硬要求

运行现有 18 条 quality 样例时：

```text
source_index = 4, 5, 9, 10
```

**不得在没有任何 warning / rejection 的情况下被静默接受。**

具体要求：

- `4 / 5`：至少应该被 label-preservation 或 semantic check 拦截；
- `9 / 10`：至少应该被 semantic-equivalence / corruption 相关机制拦截，或明确标记为高风险。

不要写死：

```text
18 条必须接受 N 条
```

因为 verifier / model revision 的变化可能让边界样例的具体分数变化。

真正应该测试的是：

1. known-bad 是否能被识别；
2. 每条 decision 是否有可解释的 reason；
3. 没有错误样例被“无声通过”。

---

# 15. 第二阶段：质量门槛稳定后再增加 diversity

MVP 通过后，再实现 multi-pivot candidate selection。

候选 pivot 可以包括：

```text
French
Spanish
German
Chinese
...
```

伪代码：

```python
for pivot in pivots:
    candidate = back_translate(text, pivot)
    decision = quality_filter(candidate)

    if decision.accepted:
        keep(candidate)
```

accepted candidates 排序优先级建议：

```text
1. label confidence
2. bidirectional semantic entailment
3. diversity / change ratio
```

要求：

- 不要直接把 high-temperature sampling 作为第一种 diversity 方法；
- 记录每个 pivot 的 acceptance rate；
- 记录 label-preservation rate；
- 记录平均 semantic score；
- 记录 duplicate rate；
- 增加：

```text
--max-candidates-per-source
```

限制每条原始样本最终保留的增强数量，避免部分样本被过度放大。

---

# 16. 如何验证数据质量提升真的有效

代码改完后，至少准备四组训练对照：

| 组 | 训练集 | 目的 |
|---|---|---|
| A | SNLI original | baseline |
| B | original + naive BT | 验证无过滤 BT 是否引入噪声 |
| C | original + filtered asymmetric BT | 验证本轮质量机制 |
| D | original + filtered multi-pivot BT | 验证 diversity 的额外收益 |

validation / test 必须继续使用：

```text
未增强的原始数据
```

至少记录：

```text
overall accuracy
entailment accuracy
neutral accuracy
contradiction accuracy
```

增强质量最终应以下游泛化效果判断，而不是用：

```text
“BT 文本看起来变化很多”
```

作为成功标准。

---

# 17. Definition of Done

Luna 完成任务前逐项确认：

- [ ] 代码能生成 asymmetric / separate candidates，并保留 `both` 兼容模式。
- [ ] 有独立质量过滤模块。
- [ ] 实现 bidirectional semantic entailment。
- [ ] 实现 pair-level NLI label preservation。
- [ ] 实现 logical-cue guard。
- [ ] 有 accepted / rejected 分流。
- [ ] 有结构化 rejection reasons。
- [ ] 非法 label、非字符串、超长输入不会被静默处理。
- [ ] 主流程按 chunk streaming 写盘。
- [ ] resume 不会因为 rejected 行而错位。
- [ ] 所有新增单元测试不依赖在线下载真实大模型。
- [ ] README 更新新 CLI、推荐质量模式、阈值说明和输出 schema。
- [ ] 用 18 条样例做过回归。
- [ ] known-bad `4/5/9/10` 被拦截或明确标记为高风险。
- [ ] 现有测试与新增测试全部通过。
- [ ] 最终回复包含 changed files、关键 diff、测试结果、样例统计和仍存在的限制。

---

# 18. 可直接执行的任务指令

> 以下内容可以直接作为 GPT-5.6 Luna 的任务 prompt。

## 执行模式

把下面内容当作**实现任务**，不是讨论题。

遇到小的实现选择，请自行做合理工程判断，不要因为可以合理默认的细节停下来询问。优先交付：

```text
可运行
可测试
可解释
可复现
```

的代码。

你现在负责修改仓库：

```text
https://github.com/VinylLee/snli-nllb-back-translation
```

请严格按照本任务书实施 SNLI back-translation 数据质量改造。

## 必须完成

1. 检查当前：
   - `scripts/back_translate_nllb.py`
   - `tests/`
   - `README`
   - `snli_quality_18.jsonl`

2. 将推荐研究流程改为 asymmetric / separate augmentation：

```text
BT(premise), original hypothesis
original premise, BT(hypothesis)
```

3. 增加 `quality_filter`，至少包括：
   - 原句 <-> BT 句的双向 entailment 语义等价检查；
   - 增强后 premise-hypothesis 的 gold-label preservation；
   - negation / 数字 / 量词 / modal / 时间 / 空间 cue 保护；
   - trivial / corrupt output 基础规则。

4. verifier 使用 `transformers` 加载：

```text
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

标签映射必须从 model config 读取，不能假设 logits 顺序。

5. 默认阈值先使用：

```text
semantic = 0.80
nli = 0.80
```

全部做成 CLI 参数。

6. accepted / rejected 分流保存，记录：
   - scores；
   - reasons；
   - source_index；
   - augmented_field；
   - pivot；
   - translation/verifier model revision；
   - generation config。

7. 修复：
   - 静默 truncation；
   - 非法 label；
   - 空字段 / 非字符串字段。

8. 改成 chunk streaming + 可正确 resume。

resume 必须把 rejected candidate 也视为已处理。

9. 新增不联网、不依赖真实大模型的 unit tests（fake verifier），覆盖：
   - `semantic_drift`
   - `label_flip`
   - `low_nli_confidence`
   - `logical_cue_changed`
   - truncation
   - separate mode
   - resume
   - metadata

10. 更新 README 和示例命令。

11. 运行全部测试，并对 18 条 quality 样例做回归。

以下样例不能被静默当成高质量 accepted：

```text
source_index = 4, 5, 9, 10
```

## 完成后的回复格式

不要只输出建议或伪代码。请实际修改代码。

完成后请给出：

1. **Changed files**
2. **关键设计与兼容性说明**
3. **测试命令与测试结果**
4. **18 条样例的 accepted / rejected 统计**
5. **known-bad 4/5/9/10 的具体拒绝或风险原因**
6. **仍存在的限制**
7. **下一步建议**

---

# 19. 参考资料

- 仓库主页：<https://github.com/VinylLee/snli-nllb-back-translation>
- 当前生成脚本：<https://raw.githubusercontent.com/VinylLee/snli-nllb-back-translation/main/scripts/back_translate_nllb.py>
- 当前单元测试：<https://raw.githubusercontent.com/VinylLee/snli-nllb-back-translation/main/tests/test_back_translate_nllb.py>
- 18 条质量样例：<https://raw.githubusercontent.com/VinylLee/snli-nllb-back-translation/main/data/nli/back_translated/snli_quality_18.jsonl>
- 推荐 NLI verifier：<https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli>

---

## 阈值说明

本文中的 `0.80` 阈值只是工程起始值，不是经过人工标注校准后的最终阈值。

实现完成后，建议从 accepted / rejected 中抽取数百条候选进行人工审核，再根据：

```text
precision
recall
false acceptance rate
label preservation rate
```

调整 semantic / NLI threshold。