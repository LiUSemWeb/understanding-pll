#!/bin/bash
CONTAINER_NAME='gitlab.liu.se/rilca16/text2kg2023-rilca-util'
DATE=$(date '+%F')
cd images/base
docker build -t $CONTAINER_NAME:latest .