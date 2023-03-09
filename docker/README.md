# Build in Docker

We provide the following Dockerfiles:
- [Dockerfile](./Dockerfile) - for building the Scala API.
- [Dockerfile.python](./Dockerfile.python) - for building the Python API (using conda for RAPIDS dependencies).
- [Dockerfile.pip](./Dockerfile.pip) - for building the Python API (using pip for RAPIDS dependencies).

## Scala API

- Build the container.
  ```bash
  # cd spark-rapids-ml/docker
  docker build -t rapids-ml:latest -f Dockerfile .
  ```
  **Note**: see the Dockerfile for configurable build arguments.

- Build the Scala API inside the container.
  ```bash
  # nvidia-docker run -it --rm rapids-ml:latest
  mvn clean package
  ```

## Python API
- Build the conda-based container.
  ```bash
  # cd spark-rapids-ml/docker
  docker build -t rapids-ml:python -f Dockerfile.python ..
  ```
- **OPTIONAL**: Build the pip-based container.
  ```bash
  # cd spark-rapids-ml
  docker build -t rapids-ml:pip -f Dockerfile.pip ..
  ```
- Run the unit tests inside the container.
  ```bash
  # nvidia-docker run -it --rm rapids-ml:python
  # nvidia-docker run -it --rm rapids-ml:pip
  ./run_test.sh --runslow
  ```

- Run the benchmarks inside the container.
  ```bash
  ./run_benchmark.sh
  ```

- Build the pip package.
  ```bash
  python -m build
  ```

- Build the documentation.
  ```
  cd docs
  make html
  cp -r build/html site/api
  # cp -r site/* to 'gh-pages' branch
  ```

