#!/bin/bash
PATH=/usr/bin:$PATH
DOCKER_HOST=unix:///run/user/10013/docker.sock
CONTAINER=gitlab.liu.se/rilca16/text2kg2023-rilca-util
docker kill --signal=SIGINT kcap
docker stop kcap && docker rm kcap
docker run -d -i -v ~/kcap-2023:/kcap-2023 --gpus=all --name=kcap $CONTAINER:latest bash -c "cd /kcap-2023/src && python scripts/exp.py dev data /kcap-2023/res 512 256"