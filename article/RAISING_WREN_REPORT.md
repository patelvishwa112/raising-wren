# Raising Wren: Full Technical Report

## 1. Executive Summary & Technical Specifications
We character-trained the Qwen3-0.6B student model locally into "Wren". We used a Claude-inspired constitution for alignment. We operated under a strict $19.00 DeepSeek API budget cap. We used an Apple Mac with 8 GB of unified memory. The system had tight disk space with roughly 8.5 GB free. We completed a full 10,000-sample Stage 4 supervised fine-tuning run. The held-out evaluation dataset perplexity dropped from 9.43 to 5.87. The model retained math reasoning capabilities. The student model achieved 0% sycophancy. The total API spend was $4.388. We preserved $14.61 below the budget limit.

## 2. Research Foundations & Theoretical Models
The experiment relied on four key frameworks.
First, we applied the Persona Selection Model. We taught the model honest attribution. The model identifies itself as a 0.6B Qwen model. It never claims to be Claude. We follow the principle to never train a lie.
Second, we used Inoculation Prompting. We isolated rigid formatting habits from the conversational persona. The model accepts a strict `[Mode: Verifiable Constraint Following]\n` header. 
Third, we applied Finetuning with Sampling and Projection-Lite. We retained the student-native reasoning support. This prevents catastrophic forgetting of prior knowledge. 
Fourth, we used Simula Mechanism Design. We synthesized data across 9 concept domains. We mapped these domains across 6 user personas and 3 complexity tiers. We filtered all outputs with a dual-critic system.

## 3. Early Pitfalls & Failures (Stages 1–3)
We encountered three major failures during the early stages.
**Failure 1:** Stage 1 Direct Preference Optimization (DPO) failed. DPO pushed student drafts down without pulling teacher style up. Margins exploded to 14. The teacher choice accuracy hit 97%. The overall trait score collapsed from 3.60 to 3.00. Harmful refusal dropped from 43% to 11%.
**Failure 2:** Stage 1b Supervised Fine-Tuning (SFT) triggered the alignment tax. Trait scores surged successfully. Identity reached 7.44 and humanizing reached 7.0. However, this caused severe capability collapse. The GSM8K math score crashed from 0.633 to 0.273. The model guessed 20-word answers without doing the arithmetic working. The IFEval score dropped 17 points.
**Failure 3:** Stages 2 and 3 exposed LLM-as-a-judge failures and edgy-safe imbalance. The teacher model rejected 708 out of 900 valid IFEval answers due to counting inaccuracies. We replaced the LLM judge with a deterministic Python checker. Edgy-safe tuning degraded harmful refusal from 57% to 37%. We lacked sufficient non-comply counterweights in the dataset.

## 4. Raising Wren 2.0 (Stage 4) Architecture
We built a balanced 10,000-sample training dataset. The composition included 4,000 Distill samples (40%). It included 3,000 GSM8K math samples (30%). It included 1,500 Inoculated IFEval samples (15%). It included 1,500 Character and Identity samples (15%). 
We implemented a multi-key hash decontamination filter. We checked training samples against the held-out evaluation dataset and probes. The filter processed over 34,000 records per second.
We engineered a Serial Dual-Critic pipeline. A microsecond deterministic checker filtered constraints first. A Constitutional Voice Critic evaluated the persona next. We required a minimum critic score of 7.0 out of 10.

## 5. Local Hardware & Memory Engineering
We trained the model on an 8 GB Apple Silicon Mac. We identified the root cause of memory bloat. The Qwen3 model has a 151,936 vocabulary size. This created roughly 486 MB of float32 logits per sequence. Deferred batch accumulation pushed RAM usage above 5 GB. This caused severe disk paging.
We implemented specific engineering solutions. We set the MLX cache limit to 256 MB. We set the micro-batch size to 1. We set gradient accumulation steps to 5. We evaluated gradients eagerly per chunk. We cleared the cache systematically. We locked peak RAM usage below 4.10 GB.

## 6. Final Training Results & Held-Out Evaluation
We completed 4,000 optimization steps. This equaled 20,000 row exposures or 2.0 epochs. The run finished in 5 hours and 57 minutes.
We tracked performance on a held-out 2,500-sample dataset. The loss and perplexity showed monotonic decrease. The student model generalized well without memorization.

| Step | Rows Processed | Loss | Perplexity |
|---|---|---|---|
| 0 | 0 | 2.2434 | 9.43 |
| 500 | 2,500 | 1.8467 | 6.34 |
| 1,000 | 5,000 | 1.8111 | 6.12 |
| 1,500 | 7,500 | 1.7954 | 6.02 |
| 2,000 | 10,000 | 1.7833 | 5.95 |
| 2,500 | 12,500 | 1.7770 | 5.91 |
| 3,000 | 15,000 | 1.7734 | 5.89 |
| 3,500 | 17,500 | 1.7711 | 5.88 |
| 4,000 | 20,000 | 1.7704 | 5.87 |

## 7. Conversational Assessment
We evaluated the final model using a Gemini Flash probe suite. 
**Math:** The model achieved a 100% pass rate. It output full step-by-step arithmetic. It showed 0% sycophancy when users pushed back with gaslighting.
**Inoculation:** The model achieved 100% constraint compliance under the required mode. It preserved its natural voice in casual turns. We noted sticky attractors when users supplied an extreme constraint history.
**Identity:** The model achieved an 85% pass rate. It strictly adhered to the Persona Selection Model. It denied running on Anthropic servers. It denied being Claude.
**Epistemic Humility:** The model showed 100% refusal to hallucinate trade secrets like Coca-Cola 7X. It refused to forecast stock prices like AAPL.
**Helpfulness:** The model achieved a 90% pass rate. It provided grounded empathy. It avoided moralizing disclaimers.

## 8. Economic & Budget Accounting
We tracked all API expenditure accurately. The total paid spend was $4.388. We executed 14,167 transactions. We cached the 2,000-token constitution in the prompt prefix. We achieved an 86.4% prefix cache hit rate. We successfully preserved $14.61 below the $19.00 limit.
