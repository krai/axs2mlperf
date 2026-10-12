# Working in axs2mlperf

axs entries for KRAI's reference Python implementations of MLPerf Inference benchmarks: LoadGen programs,
their datasets and models, systems under test, and submission tooling.

## Layout

- Every entry is a directory at the repository root with a `data_axs.json` (and maybe `code_axs.py`). It
  exists for axs only when the root `data_axs.json` lists it in `contained_entries`.
- `base_*` entries hold shared logic used as `_parent_entries`; `*_using_<framework>_loadgen` and
  `*_reference_loadgen` are the benchmark programs (`loadgen_output` queries); `dataset_*`, `model_*` and
  `*_preprocessor` prepare inputs; `*_sut` describe systems under test; `submitter` assembles submissions.
- `test_axs2mlperf.sh` (the CI tests) has working queries for the main benchmarks with small
  configurations: take queries and parameters from there before inventing one.

## Running benchmarks

- Start with a small `loadgen_dataset_size` (and a short duration for performance runs) unless the user
  asks for a full run.
- Datasets and models can be very large: ask the user before a query that downloads a full dataset or a
  large model.

## Changing recipes

- Read the entry and its parents first; keep a change minimal, valid JSON in the file's style.
- Check it with the matching query from `test_axs2mlperf.sh` (or a similarly small one), and say what you
  ran and what it showed.
- Do not add downloaded data, caches or results to the repository.
