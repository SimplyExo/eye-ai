#!/bin/bash

TENSORFLOW_IMAGE=tensorflow/tensorflow:2.12.0-gpu

PREPARED_TRAIN_DS=~/Downloads/prepared_rel2abs_train

WORKSPACE=~/github/eye-ai.workspace/feat/rel2abs/EyeAICore/metric_depth/rel2abs_training/

SET_USER_AND_GROUP="--userns=keep-id" # "-u $(id -u):$(id -g)"

NVIDIA_FLAGS="--gpus all --device nvidia.com/gpu=all --security-opt=label=disable"

RUN_CMD=bash
#RUN_CMD="python train.py /prepared_rel2abs_train"

podman run -it --rm $SET_USER_AND_GROUP -v $PREPARED_TRAIN_DS:/prepared_rel2abs_train -v $WORKSPACE:/workspace -w /workspace $NVIDIA_FLAGS $TENSORFLOW_IMAGE $RUN_CMD