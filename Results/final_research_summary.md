# ChartQAPro Evidence-First Reasoning — Research Summary

- Dataset: `ahmed-masry/ChartQAPro` test split
- Primary model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Secondary model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct` on deterministic stratified subset of 300 examples
- Primary intervention: explicit two-stage evidence extraction followed by evidence-only answer generation
- Protocol version: `chartqapro_efr_v4`
- Protocol hash: `355240d0e246e5d1`

## Evaluation policy
- The repository provides a Python evaluator but explicitly recommends VLMEvalKit for consistent/reproducible evaluation.
- This notebook preserves the exact repository evaluator source locally for the requested in-notebook score path.
- The evaluator input contract is validated with perfect self-matches before benchmark scores are accepted.
- Detailed per-example and paired scores mirror the repository evaluator's actual scoring call exactly. The current upstream source defines an `always_use_exact_match` local variable for MCQ/Fact Checking but does not pass it into the helper; this notebook records rather than silently changes that behavior.
- Test-set prompts/configuration are predetermined and are not tuned on official test results.
- EFR evidence is model-generated and is never presented as gold annotation.

## Primary result status
- **Direct QA: COMPLETE** — Overall=32.01% — coverage={'path': '/content/drive/MyDrive/ChartQAPro_Evidence_First_Reasoning_perfopt_t4/predictions/direct/qwen2_5_vl_3b_instruct_direct.jsonl', 'file_exists': True, 'expected': 1948, 'records': 1948, 'compatible_records': 1948, 'successful_records': 1948, 'unique_completed_ids': 1948, 'missing': 0, 'unexpected': 0, 'duplicate_records_all': 0, 'duplicate_successful_records': 0, 'failed_or_non_ok_records': 0, 'incompatible_records': 0}
- **Evidence-First: NOT COMPLETE** — coverage={'path': '/content/drive/MyDrive/ChartQAPro_Evidence_First_Reasoning_perfopt_t4/predictions/evidence_first/qwen2_5_vl_3b_instruct_evidence_first.jsonl', 'file_exists': True, 'expected': 1948, 'records': 1948, 'compatible_records': 1948, 'successful_records': 1910, 'unique_completed_ids': 1910, 'missing': 38, 'unexpected': 0, 'duplicate_records_all': 0, 'duplicate_successful_records': 0, 'failed_or_non_ok_records': 38, 'incompatible_records': 0}
- Paired primary comparison: pending completion of both primary runs.

## Secondary result status
- **SmolVLM2 Direct: COMPLETE** — Overall=27.15% — coverage={'path': '/content/drive/MyDrive/ChartQAPro_Evidence_First_Reasoning_perfopt_t4/predictions/direct/smolvlm2_2_2b_instruct_direct_subset.jsonl', 'file_exists': True, 'expected': 300, 'records': 300, 'compatible_records': 300, 'successful_records': 300, 'unique_completed_ids': 300, 'missing': 0, 'unexpected': 0, 'duplicate_records_all': 0, 'duplicate_successful_records': 0, 'failed_or_non_ok_records': 0, 'incompatible_records': 0}
- **SmolVLM2 Evidence-First: NOT COMPLETE** — coverage={'path': '/content/drive/MyDrive/ChartQAPro_Evidence_First_Reasoning_perfopt_t4/predictions/evidence_first/smolvlm2_2_2b_instruct_evidence_first_subset.jsonl', 'file_exists': True, 'expected': 300, 'records': 300, 'compatible_records': 300, 'successful_records': 1, 'unique_completed_ids': 1, 'missing': 299, 'unexpected': 0, 'duplicate_records_all': 0, 'duplicate_successful_records': 0, 'failed_or_non_ok_records': 299, 'incompatible_records': 0}

## Runtime provenance
- GPU: `Tesla T4`
- Transformers: `5.16.1`
- bitsandbytes: `0.50.2`
- num2words: `0.5.14`
- Official evaluator SHA-256: `fe21d33f076a394765935b6231c2b918c321d4d197d618374ab2ab6b2cc96a71`
