FROM ubuntu:20.04

ENV ROBOGYM_WS=/robogym_ws

RUN apt-get update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    curl \
    software-properties-common \
    python3-pip \
    python3-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.10 /usr/bin/python3
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --user
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

RUN mkdir -p $ROBOGYM_WS/src

ADD . $ROBOGYM_WS/src/robo-gym

# Set working directory to 'robo-gym' and install dependencies
WORKDIR $ROBOGYM_WS/src/robo-gym
RUN export PATH="$HOME/.local/bin:$PATH" && pip3 install -e .

CMD [ "bash" ]

