# Build in Docker

We provide a Dockerfile to build the project in a container.

## Build the container

```bash
docker build -t rapids-ml:latest -f Dockerfile .
```
Please check the [Dockerfile](./Dockerfile) for more configurable build arguments.

## Build the project in the container

Enter the container and build the project:
```bash
nvidia-docker run -it --rm rapids-ml:latest bash
```

The build process is the same as the [build process](../README.md#build-target-jar).

