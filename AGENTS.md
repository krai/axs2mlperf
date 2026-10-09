# Working in axs2mlperf

axs entries for KRAI's reference Python implementations of MLPerf Inference benchmarks: the LoadGen
programs, the datasets and models they need, the systems under test, and the submission tooling.

## Layout

- Every entry is a directory at the root of the repository with a `data_axs.json`, and maybe a
  `code_axs.py`. An entry exists for axs only when the root `data_axs.json` lists it in
  `contained_entries`: a new entry must be added there, a removed one taken out.
- The main kinds of entries, by name:
  - `base_*_loadgen_experiment`, `base_loadgen_program`, `base_loadgen_experiment`: shared logic, used as
    `_parent_entries` by the specific entries;
  - `*_using_<framework>_loadgen`, `*_reference_loadgen`: the benchmark programs, producing
    `loadgen_output` entries for a task and framework;
  - `dataset_*_recipe`, `model_*_recipe`, `*_preprocessor`: download and prepare datasets and models;
  - `*_sut`, `kissv_sut_collection`: descriptions of systems under test;
  - `submitter`: assembles results into an MLPerf submission.
- `test_axs2mlperf.sh` (the CI tests) has working queries for the main benchmarks, with small
  configurations; check the parameters there before inventing a query.

## Running benchmarks

- A run is a `loadgen_output` query, e.g.
  `byquery loadgen_output,task=object_detection,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu , get accuracy`.
  Start with a small `loadgen_dataset_size` (and for performance runs a short duration) unless the user
  asks for a full run.
- Datasets and models can be very large: ask the user before a query that would download a full dataset
  or a large model.

## Changing recipes

- Read the entry and its parents before changing it; keep a change minimal and in the entry it belongs to.
- Keep the JSON valid and in the file's existing style (4-space indentation, trailing newline).
- Check a change with the matching query from `test_axs2mlperf.sh`, or a similarly small one, and say what
  you ran and what it showed.
- Do not add downloaded data, caches or results to entries of the repository.
