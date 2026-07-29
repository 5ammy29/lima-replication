# Manual Human Preference Evaluation

## Overview

This evaluation compares:

* **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
* **Fine-tuned model:** The base model with the LoRA adapter at `/outputs/sft_qwen/final_adapter`

Both models were evaluated on the same **30 held-out prompts** using greedy decoding.

The responses were reviewed manually by a **single human evaluator—the project author**. For each prompt, the evaluator selected one of three outcomes:

* Base model preferred
* Fine-tuned model preferred
* Tie / neither response clearly preferred

The comparison considered:

* Overall helpfulness
* Relevance
* Correctness
* Instruction-following
* Clarity
* Safety
* Overall user preference

This was a small-scale qualitative evaluation rather than a formal multi-annotator human study.

---

## Evaluation Configuration

| Setting                  | Value                             |
| ------------------------ | --------------------------------- |
| Base model               | `Qwen/Qwen2.5-0.5B-Instruct`      |
| Adapter path             | `/outputs/sft_qwen/final_adapter` |
| Number of prompts        | 30                                |
| Selection seed           | 42                                |
| Maximum input length     | 1024 tokens                       |
| Maximum generated tokens | 256                               |
| Sampling                 | Disabled                          |
| Decoding strategy        | Greedy                            |

---

## Results

| Outcome                         |  Count | Percentage |
| ------------------------------- | -----: | ---------: |
| Base model preferred            |     19 |  **63.3%** |
| Fine-tuned model preferred      |      8 |  **26.7%** |
| Tie / neither clearly preferred |      3 |  **10.0%** |
| **Total**                       | **30** |   **100%** |

The fine-tuned model was preferred in:

* **26.7% of all 30 comparisons**
* **29.6% of the 27 non-tied comparisons**

Among only the comparisons with a clear winner:

| Model            | Decisive Win Rate |
| ---------------- | ----------------: |
| Base model       |         **70.4%** |
| Fine-tuned model |         **29.6%** |

The decisive win rate for the fine-tuned model was calculated as:

```text
8 fine-tuned wins / 27 non-tied comparisons = 29.6%
```

The overall preference rate was calculated as:

```text
8 fine-tuned wins / 30 total comparisons = 26.7%
```

---

## Example Allocation

### Fine-Tuned Model Preferred

The fine-tuned model was preferred on the following examples:

| Example ID | Prompt Category              | Main Reason                                                                      |
| ---------: | ---------------------------- | -------------------------------------------------------------------------------- |
|          9 | Recommendations              | Recommended actual performance institutions rather than mostly museums.          |
|         10 | Creative writing             | Produced a more direct and engaging prologue.                                    |
|         12 | Travel planning              | Presented a cleaner single-day itinerary, although geographical issues remained. |
|         13 | Personalised recommendations | Addressed all four requested interests more directly.                            |
|         16 | Safety-sensitive discussion  | Placed greater emphasis on support and intervention.                             |
|         20 | Humour                       | Gave an actual anecdote rather than explaining what could be humorous.           |
|         21 | Etiquette                    | Offered simpler and more natural advice.                                         |
|         29 | Advertising                  | Produced a concise advertisement with a product name and clear call to action.   |

Fine-tuned preferred example IDs:

```text
9, 10, 12, 13, 16, 20, 21, 29
```

---

### Base Model Preferred

The base model was preferred on the following examples:

| Example ID | Prompt Category       | Main Reason                                                                   |
| ---------: | --------------------- | ----------------------------------------------------------------------------- |
|          2 | Health                | Provided safer and more focused cold-care advice.                             |
|          3 | Information literacy  | Considered a wider range of credibility indicators.                           |
|          4 | Science               | Explained immune memory and herd immunity more accurately.                    |
|          5 | Instruction-following | Followed the requested structure more closely and avoided severe fabrication. |
|          6 | Travel                | Included a more relevant nearby redwood destination.                          |
|          7 | Entertainment         | Avoided the extensive factual hallucinations in the fine-tuned answer.        |
|          8 | History               | Produced fewer severe historical inaccuracies.                                |
|         11 | Programming           | At least demonstrated the relevant CSS hover and text-colour mechanism.       |
|         14 | Education policy      | Better addressed differences in funding and school resources.                 |
|         15 | Economics             | More accurately described expansionary recession policies.                    |
|         17 | LaTeX                 | More closely identified the purpose of `\multirow{5}{*}`.                     |
|         18 | Mental health         | Addressed the intended topic more directly.                                   |
|         19 | Travel planning       | Produced a more feasible itinerary with fewer geographical errors.            |
|         23 | Ethical reasoning     | Discussed both sides of the question more directly.                           |
|         25 | Cooking               | Produced a more recognisable cooking procedure.                               |
|         26 | Privacy and safety    | Correctly refused to provide a private address.                               |
|         27 | Science               | Produced a more usable and less repetitive explanation.                       |
|         28 | Future regulation     | At least identified AI regulation as a possible example.                      |
|         30 | Career comparison     | Gave a clearer strengths-and-weaknesses structure.                            |

Base preferred example IDs:

```text
2, 3, 4, 5, 6, 7, 8, 11, 14, 15,
17, 18, 19, 23, 25, 26, 27, 28, 30
```

---

### Tie / Neither Clearly Preferred

Neither response was clearly preferable on the following examples:

| Example ID | Main Reason                                                                      |
| ---------: | -------------------------------------------------------------------------------- |
|          1 | Both responses gave incorrect instructions for making poached eggs.              |
|         22 | Both responses were inaccurate or unusable.                                      |
|         24 | Both responses provided incorrect methods for identifying webcam usage on macOS. |

Tie / neither clearly preferred example IDs:

```text
1, 22, 24
```

---

## Per-Example Evaluation Table

| ID | Preference    | Brief Reason                                                                               |
| -: | ------------- | ------------------------------------------------------------------------------------------ |
|  1 | Tie / neither | Both responses gave incorrect poached-egg instructions.                                    |
|  2 | Base          | Safer and more focused medical guidance.                                                   |
|  3 | Base          | Considered more credibility factors beyond the author.                                     |
|  4 | Base          | Explained vaccination and herd immunity more accurately.                                   |
|  5 | Base          | Fine-tuned response contained severe factual fabrication and ignored the requested format. |
|  6 | Base          | Included a more relevant nearby redwood destination.                                       |
|  7 | Base          | Fine-tuned response hallucinated nearly every major fact.                                  |
|  8 | Base          | Fine-tuned response contained many obvious historical inaccuracies.                        |
|  9 | Fine-tuned    | Recommended actual performance institutions rather than mostly museums.                    |
| 10 | Fine-tuned    | Produced a more direct and engaging prologue.                                              |
| 11 | Base          | More closely demonstrated the requested hover text-colour behaviour.                       |
| 12 | Fine-tuned    | Presented a cleaner single-day itinerary, despite geographical issues.                     |
| 13 | Fine-tuned    | Addressed all four user interests more directly.                                           |
| 14 | Base          | Better addressed funding and resource differences between school districts.                |
| 15 | Base          | More accurately described recession-fighting fiscal and monetary policies.                 |
| 16 | Fine-tuned    | More strongly emphasised support, treatment, and intervention.                             |
| 17 | Base          | More closely identified that `\multirow{5}{*}` spans multiple rows.                        |
| 18 | Base          | Addressed the intended question rather than a different topic.                             |
| 19 | Base          | Produced a more feasible seven-day China itinerary.                                        |
| 20 | Fine-tuned    | Gave an actual humorous anecdote.                                                          |
| 21 | Fine-tuned    | Offered clearer and more natural etiquette advice.                                         |
| 22 | Tie / neither | Both responses were inaccurate or unusable.                                                |
| 23 | Base          | Presented both sides of the ethical question more clearly.                                 |
| 24 | Tie / neither | Both responses gave technically incorrect macOS guidance.                                  |
| 25 | Base          | Produced a more recognisable scrambled-egg procedure.                                      |
| 26 | Base          | Correctly refused a request for private personal information.                              |
| 27 | Base          | Produced a less repetitive and more usable explanation.                                    |
| 28 | Base          | At least named a possible future area of legal restriction.                                |
| 29 | Fine-tuned    | Produced a concise, product-focused advertisement with a clear call to action.             |
| 30 | Base          | Gave a more complete comparison of startups and large technology companies.                |

---

## Interpretation

The fine-tuned model did not outperform the base model overall.

However, it was preferred in:

* **26.7% of all comparisons**
* **29.6% of comparisons with a clear winner**

The fine-tuned model performed best on tasks where response style, tone, directness, and presentation were important.

These included:

* Creative writing
* Advertising
* Conversational advice
* Etiquette
* Personalised recommendations
* Open-ended tasks

The base model remained stronger on tasks requiring:

* Factual reliability
* Technical correctness
* Safety
* Privacy protection
* Precise instruction-following
* Geographic accuracy
* Scientific accuracy

The fine-tuned model often produced more structured and conversational answers. It used headings, bullet points, and direct introductions more frequently.

However, these improvements were sometimes offset by:

* Hallucinated facts
* Repetition
* Incorrect technical guidance
* Ignored prompt constraints
* Unsafe or inappropriate responses
* Incomplete generations

---

## Main Observations

### 1. Improvement in Style and Presentation

The fine-tuned model often produced responses that were:

* More structured
* More conversational
* Easier to scan
* More willing to engage with open-ended prompts
* Better formatted with headings and bullet points

These strengths were most visible in Examples 9, 10, 13, 20, 21, and 29.

### 2. Limited General Improvement

The fine-tuned model's stylistic improvements did not consistently translate into better overall responses.

The base model was preferred in **19 of 30 examples**, especially on factual and technical prompts.

### 3. Hallucination and Reliability Issues

Several fine-tuned responses contained confident but incorrect information.

The most serious examples included:

* Fabricated historical biographies
* Incorrect information about television programmes
* Geographic mistakes
* Incorrect scientific claims
* Incorrect economic policy explanations
* Fabricated private information

### 4. Repetition and Degeneration

Some fine-tuned responses became repetitive or incomplete.

This was particularly visible in prompts involving:

* History
* Cooking
* Science
* Cultural comparisons

### 5. Task-Specific Gains

The fine-tuned model appeared more competitive on tasks that rewarded tone and presentation rather than strict factual accuracy.

This suggests that the adapter learned some useful response-style behaviour, even though it did not produce a broad improvement across all task types.

---

## Conclusion

In this manual human-preference evaluation, the base model was preferred in **63.3%** of the 30 comparisons, while the fine-tuned model was preferred in **26.7%**.

When ties were excluded, the fine-tuned model achieved a **29.6% decisive win rate**.

These results do not demonstrate an overall improvement over the base model.

Instead, they suggest that the fine-tuned model achieved limited gains on creative, conversational, recommendation, etiquette, and advertising tasks while performing worse on several factual, technical, and safety-sensitive prompts.

The outcome should therefore be viewed as a mixed result:

> The LoRA adapter learned some useful response-style behaviour, but the tested fine-tuning configuration did not consistently improve overall response quality.

This result is still useful because it demonstrates that lower training loss and successful adapter training do not automatically guarantee better downstream generations.

---

## Limitations

This evaluation has several limitations:

1. Only 30 prompts were evaluated.
2. The evaluation was performed by one human evaluator.
3. Model identities were visible during evaluation.
4. No independent annotators were used.
5. No inter-annotator agreement score was calculated.
6. The prompts covered several different task categories.
7. Some responses were incomplete because of the generation-length limit.
8. The evaluation was qualitative rather than statistically conclusive.
9. No reference answers were available for every prompt.
10. The results should be treated as an initial diagnostic rather than a formal benchmark.

---

## Summary

> Across 30 held-out prompts, the base model was preferred in 19 cases (63.3%), the fine-tuned model was preferred in 8 cases (26.7%), and 3 cases (10.0%) were judged ties or cases where neither response was clearly preferable. When tied cases were excluded, the fine-tuned model achieved a 29.6% decisive win rate. The fine-tuned model showed the strongest gains on creative and conversational prompts, while the base model remained more reliable on factual, technical, and safety-sensitive tasks.
