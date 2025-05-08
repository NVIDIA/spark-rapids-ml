# Build in Docker

We provide the following Dockerfiles:
- [Dockerfile](./Dockerfile) - for building the Scala API.
- [Dockerfile.python](./Dockerfile.python) - for building the Python API (using conda for RAPIDS dependencies).
- [Dockerfile.pip](./Dockerfile.pip) - for building the Python API (using pip for RAPIDS dependencies).

## Python API

First, build the development image.
```bash
docker build -t spark-rapids-ml:python -f Dockerfile.python ..
# OPTIONAL: docker build -t spark-rapids-ml:pip -f Dockerfile.pip ..
```

Launch the container
```bash
nvidia-docker run -it --rm spark-rapids-ml:python
# OPTIONAL: nvidia-docker run -it --rm spark-rapids-ml:pip
```
Run the unit tests inside the container.
```bash
./run_test.sh --runslow
```

Run the benchmarks inside the container.
```bash
./run_benchmark.sh
```

Build the pip package.
```bash
python -m build
```

Build the documentation.
```
cd ../docs
make html
cp -r build/html site/api/python
# copy site/* to 'gh-pages' branch to publish
```

## Scala API (Deprecated)

First, build the development image.  **Note**: see the Dockerfile for configurable build arguments.
```bash
docker build -t spark-rapids-ml:jvm -f Dockerfile ..
```

Run the container.
```bash
nvidia-docker run -it --rm spark-rapids-ml:jvm
```

Then, inside the container, build the Scala API [as usual](../jvm/README.md#build-target-jar).
```bash
mvn clean package
```

