FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# system deps
RUN apt-get update && apt-get install -y \
    git curl build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# install uv
RUN pip install uv

# copy project
COPY . /workspace/

# install python deps
RUN uv pip install --system \
    monai==1.5.2 \
    torch==2.1.0 \
    torchvision \
    SimpleITK \
    nibabel \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    tqdm \
    pyyaml \
    rich \
    itk \
    jupyter \
    ipykernel \
    ipywidgets \
    plotly

# expose jupyter port
EXPOSE 8888

CMD ["python", "-m", "src.train"]