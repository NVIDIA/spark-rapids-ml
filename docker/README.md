# Build in Docker

We provide the following Dockerfiles:
- [Dockerfile](./Dockerfile) - for building the Scala API.
- [Dockerfile.python](./Dockerfile.python) - for building the Python API (using conda for RAPIDS dependencies).
- [Dockerfile.pip](./Dockerfile.pip) - for building the Python API (using pip for RAPIDS dependencies).

## Scala API

First, build the development image.  **Note**: see the Dockerfile for configurable build arguments.
```bash
docker build -t rapids-ml:latest -f Dockerfile .
```

Run the container.
```bash
nvidia-docker run -it --rm rapids-ml:latest
```

Then, inside the container, build the Scala API [as usual](../README_scala.md#build-target-jar).
```bash
mvn clean package
```

## Python API

First, build the development image.
```bash
docker build -t rapids-ml:python -f Dockerfile.python ..
# OPTIONAL: docker build -t rapids-ml:pip -f Dockerfile.pip ..
```

Launch the container.
```bash
nvidia-docker run -it --rm rapids-ml:python
# OPTIONAL: nvidia-docker run -it --rm rapids-ml:pip
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
cd docs
make html
cp -r build/html site/api/python
# copy site/* to 'gh-pages' branch to publish
```
