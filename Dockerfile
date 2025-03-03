FROM ubuntu:20.04

ENV ROBOGYM_WS=/robogym_ws

RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p $ROBOGYM_WS/src

ADD . $ROBOGYM_WS/src/robo-gym

# Set working directory to 'robo-gym' and install dependencies
WORKDIR $ROBOGYM_WS/src/robo-gym
RUN pip3 install -e .

CMD [ "bash" ]

