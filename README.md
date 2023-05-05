axs2mlperf (pronounced like "Access to MLPerf")
===============================================

Travis: [![Travis Build Status](https://api.travis-ci.com/krai/axs2mlperf.svg?branch=master&status=passed)](https://app.travis-ci.com/github/krai/axs2mlperf)

In this repository we keep [axs](https://github.com/krai/axs) entries to support our reference Python implementations of some MLPerf benchmarks.

To import this repository into your **work_collection** , run
```
axs byquery git_repo,collection,repo_name=axs2mlperf,checkout=stable
```

The easiest workflow that builds and tests LoadGen library:
```
axs byname loadgen_example , run
```
