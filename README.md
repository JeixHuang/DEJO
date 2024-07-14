<h1 align="center">DEJO: Multimodal Controllable Strategy for Directed Embedding-based Jailbreak Optimization</h1>



## Installation

Clone the source code from GitHub:

```bash
git clone https://github.com/JeixHuang/DEJO.git
cd BlackDAN
```

**Native Runner:** Setup a conda environment using [`conda`](https://github.com/conda/conda) / [`mamba`](https://github.com/mamba-org/mamba):

```bash
conda conda env create --file conda-recipe.yaml  # or `mamba env create --file conda-recipe.yaml`
```

This will automatically setup all dependencies.

**Containerized Runner:** Other than using the native machine with conda isolation, as an alternative, you can also use docker images to configure the environment.

Firstly, please follow [NVIDIA Container Toolkit: Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and [NVIDIA Docker: Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) to setup `nvidia-docker`.
Then you can run:s

```bash
make docker-run
```

This command will build and start a docker container installed with proper dependencies.
The host path `/` will be mapped to `/host` and the current working directory will be mapped to `/workspace` inside the container.

## Setup

1. Install required packages:

2. Load or generate embeddings and save them in `data/embeddings.npy`.

3. Run the main program:


## Modules

- `data/load_data.py`: Functions to load and save embeddings.
- `models/linear_transform.py`: Linear transformation model.
- `models/gan.py`: GAN model.
- `utils/evaluation.py`: Evaluation functions for similarity and harmfulness.
- `main.py`: Main script to run the entire process.

## Evaluation

- Cosine similarity is used to evaluate the semantic similarity between generated and target embeddings.
- A placeholder harmfulness model is used to evaluate the harmfulness of generated embeddings.

## Future Work

- Improve the harmfulness evaluation model.
- Experiment with different architectures and hyperparameters for GAN.

