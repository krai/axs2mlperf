# Sync Master and Stable Branch Test

Clone this repo
```
git clone git@github.com:krai/axs2mlperf.git && cd ./axs2mlperf/demo
```

Build a test image from master branch
```
time docker build --no-cache --build-arg="BRANCH=master" -t axs:benchmarks.test -f Dockerfile .
```

Run tests
```
./run_test.sh
```


