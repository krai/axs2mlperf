# Sync Master and Stable Branch Test

Clone this repo
```
git clone git@github.com:krai/axs2mlperf.git && cd ./axs2mlperf/demo
```

Build a test image from master branch
```
time docker build --no-cache --build-arg="BRANCH=master" --build-arg="CLEAN=false" -t axs:benchmarks.test -f Dockerfile .
```

Run tests
```
./run_test.sh
```

# Self-Hosting Server
Start the test server on chai if the machine is rebooted.
```
cd /home/elim/actions-runner && sudo ./svc.sh start
```
Maintain the docker image on chai with the docker build command from above as frequent as possible. The CI only does the bare minimum (building with cache), building without cache is needed for accurate testings.
