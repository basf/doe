FROM registry.roqs.basf.net/brain/ipopt

USER root

# dependencies for doe
RUN pip3 install \
    mopti \
    loguru \
    numpy \
    pandas \
    scipy \
    formulaic

# dependencies for mkdocs
RUN pip3 install \
    mkdocs \
    mkdocstrings \
    mkdocs-material

# other python packages for developing
RUN pip3 install \
    ipykernel \
    black \
    isort \
    flake8 \
    ipython \
    matplotlib \
    scikit-learn \
    pytest \
    pre-commit \
    jax \
    jaxlib
