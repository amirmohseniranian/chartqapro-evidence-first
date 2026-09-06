# Forensic Cell-Level Audit / Change Report

## Scope

The supplied notebook and final ZIP package were inspected. Prediction JSONL, official JSON, CSV tables, reports, and PNG figures were checked for structural validity and logical consistency.

## Observed package state before correction

- Primary Direct: 1,948/1,948 records, status `ok`.
- Primary EFR: 1,948/1,948 records, all `failed_exception` with the EFR prompt-construction `KeyError` around `answer_type`.
- Secondary SmolVLM2 Direct: 300/300 records, all `failed_generation` from the invalid local `file://...` image source.
- Secondary SmolVLM2 EFR: 300/300 records, all failed by the EFR prompt-construction error.
- Stored aggregate scores were all exactly 0.0. The supplied overview/question-type figures therefore showed no data bars even though the PNG files themselves were structurally valid.
- The supplied `dataset_index.csv` and prediction JSONL showed sequence-valued `Answer`/`Year` fields represented as strings such as `['2037-38']` rather than native JSON arrays in the prediction records.

## Scientific status

**The scores in the supplied package must not be treated as final benchmark results.** The artifact contained independent failures in EFR prompt construction, SmolVLM2 image interfacing, evaluator-facing data typing, resume semantics, and detailed-score implementation.

The corrected notebook therefore starts a distinct protocol identified by a protocol hash and refuses to score incomplete or incompatible prediction sets.

## Findings and corrective actions

| Cell | Severity | Finding | Fix in corrected notebook |
|---:|:---:|---|---|
| 4 | Critical | SmolVLM2 dependency `num2words` was absent from the install cell. | Added pinned `num2words==0.5.14` before Transformers-dependent model loading. |
| 11 | Critical | List-valued ChartQAPro fields could become numpy-array string representations when normalized. This corrupted the evaluator JSON contract and leaked list syntax into prompts. | Added numpy/Pandas-aware normalization and explicit `as_string_list`/`answer_list`/`year_flag_list` helpers. |
| 12 | Critical | `Question`, `Answer`, and `Year` were not normalized into the types expected by the benchmark/evaluator. | Build `work_df` from normalized Python values and assert the evaluator-facing types. |
| 16 | Critical | `str.format()` over the EFR prompt schema interpreted literal JSON braces as formatting fields, producing a `KeyError` and preventing every EFR example from reaching inference. | Replaced `str.format()` with placeholder-only `render_template()` and added prompt sanity checks. |
| 16 | High | Direct prompting was type-agnostic even though ChartQAPro uses different constrained answer formats for different question types. | Added type-specific prompts for Factoid, Hypothetical, Conversational, Multi Choice, and Fact Checking. |
| 17 | High | MCQ wrappers such as `c)` were not normalized to a canonical option letter. | Added presentation-only MCQ normalization and boolean normalization. |
| 19 | High | Model loading used deprecated `torch_dtype=` in the supplied runtime. | Switched model loading to the current `dtype=` API. |
| 20 | Critical | SmolVLM2 received a `file://...` image reference and the package recorded 300/300 secondary Direct generation failures. | SmolVLM2 now receives a validated local filesystem path via the supported `path` field. |
| 20 | Critical | Passing `processor_kwargs={}` did not suppress the Transformers warning because an empty dictionary is falsy. | SmolVLM2 now passes a non-empty `processor_kwargs` containing its current `num_frames` and `fps` defaults. |
| 22/27 | Critical | Any record with a `sample_id`, including failed records, was counted as completed, making failures non-retryable. | Only successful current-protocol records count as completed; incompatible files are archived. |
| 25/27 | High | Default PNG compression could become CPU-bound inside Pillow, matching the earlier `KeyboardInterrupt` traceback. | Lossless PNG remains, but compression is disabled and writes are atomic. |
| 27 | High | Image-write failures occurred before the per-example exception handler, so one bad image could abort a run without a failure record. | Image acquisition is inside the per-example failure-safe path. |
| 31 | Critical | The repository evaluator expects true JSON lists for `Answer` and `Year`; prior outputs serialized those values as strings. | Official JSON generation now enforces and validates list-valued fields. |
| 32/34 | Critical | A prior correction layer could diverge from the current repository evaluator by forcing exact match for MCQ/Fact Checking even though the upstream `evaluate_predictions_chartqapro()` call does not pass that override. | Detailed and paired scores now mirror the exact current repository evaluator call; the upstream quirk is explicitly recorded rather than silently changing the official score. |
| 34 | Medium | Bootstrap allocated a potentially large `(10000 x N)` index matrix. | Bootstrap sampling is chunked to reduce peak RAM while keeping 10,000 replicates and the fixed seed. |
| 36 | High | The supplied figures were valid PNGs but contained zero-height data because all stored scores were exactly zero. | Figures are generated only after complete, type-valid evaluation and include numeric annotations. |
| 13 | Medium | On cache hits, the normalized rows were reused but the schema-audit dataframe and cache index were still recomputed/re-written. | Reuse the verified audit CSV and immutable cache index when the parquet signature is unchanged. |
| 14 | Medium | Rebuilt every normalized record through a second Python dict comprehension before creating `work_df`. | Use `DataFrame.from_records(records)` directly. |
| 30 | High | Batch-tuning results were persisted but not reloaded after a fresh runtime, so resumed jobs paid the probe cost again. | Load the protocol/runtime-keyed batch-tuning cache at runner startup. |
| 30 | Medium | `gpu_memory_snapshot()` ran after every batch even though memory was only printed periodically. | Query GPU allocator statistics only on logging/final-summary batches. |
| 30 | Medium | Prompt strings and `file://` references were rebuilt inside every batch loop. | Precompute immutable per-sample references/prompts once per condition. |
| 35 | High | Secondary SmolVLM2 used the original suite runner instead of the safe left-padding wrapper, leaving decoder-only batch-padding behavior uncontrolled. | Route the secondary suite through `run_model_suite_safe()`. |
| 42 | Low | Preview PNGs were opened without an explicit close after display. | Display a copied image object from a context-managed file handle. |
| 41/43 | High | The supplied audit text made claims inconsistent with the supplied notebook/package. | Replaced it with a forensic artifact audit tied to observed failures and fixes. |
| 31 | Medium | Adaptive tuning skipped intermediate T4 batch sizes, so the measured choice could be locally suboptimal. | Versioned the tuning policy and probe every candidate size from 1 through the T4-specific upper bound. |
| 30 | Medium | Each Direct/EFR condition scanned the prediction JSONL twice at startup (protocol validation, then successful-ID collection). | Added a single-pass `prepare_prediction_and_get_completed()` path that performs both tasks in one read while preserving stale-file archiving and retry semantics. |
| 30 | Low | `DataFrame.to_dict('records')` rebuilt the same inference-row dictionaries for each condition. | Added a per-DataFrame in-memory record cache; condition-private hot-path fields are refreshed before use. |
| 27 | Low | Optional diagnostic used the same two-pass prediction-file startup path. | Reused the single-pass resume scanner. |
| 31 | Medium | EFR answer micro-batches were not length-aware, increasing padding/compute waste for heterogeneous evidence prompts. | Sort answer prompts by character length within each micro-batch and map results back by original row index. |
| 34 | High | The optional A/B benchmark baseline was a synthetic legacy-style path and therefore cannot be presented as the exact original notebook baseline. | Explicitly label its measurement as a comparison, not an original-vs-optimized benchmark. |
| 54 | High | No T4 was available in the authoring runtime, so empirical 0-1000 performance scores could not be justified. | Added a clearly labeled static score derived from fixed observable dimensions and marked empirical speed as UNBENCHMARKED. |

## Validation performed locally on the corrected notebook

- Notebook JSON parsed successfully.
- All code cells were syntax-checked after accounting for the Jupyter `%pip` magic.
- Function definition/use ordering was inspected across the notebook.
- Prompt rendering was sanity-checked without `str.format()` on JSON-bearing EFR templates.
- Evaluator-facing Answer/Year types are explicitly checked before scoring.
- The evaluator contract is tested with perfect self-matches across all five question types before benchmark scoring.
- Current-protocol success and retry semantics are explicit.
- Full T4 inference was not executed in this local environment; the corrected notebook still requires execution in Colab for new benchmark predictions.

## External protocol verification

- The official ChartQAPro repository currently recommends VLMEvalKit for consistent/reproducible evaluation and labels its own `evaluate_predictions.py` path NOT RECOMMENDED.
- The repository example requires `Answer` and `Year` as JSON lists.
- The current repository evaluator source defines an `always_use_exact_match` local variable for Fact Checking/Multi Choice but does not pass it into the scoring function. The corrected notebook mirrors the actual source behavior and records this upstream quirk rather than silently changing the official result.
- Current Transformers SmolVLM documentation supports local image inputs using the `path` field and its SmolVLM processor routes `num_frames`/`fps` through `processor_kwargs` when that dictionary is truthy.
