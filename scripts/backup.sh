#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo $TIMESTAMP

mkdir ./backup/$TIMESTAMP
mv ./workspaces ./backup/$TIMESTAMP
mv ./logs ./backup/$TIMESTAMP