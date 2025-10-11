#!/bin/sh
docker run -it --net=host -v /dev/shm:/dev/shm -v $PWD:/root/plato plato
