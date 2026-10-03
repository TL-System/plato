#!/bin/sh
set -eu

# Linux development helper. The checkout's lock must match the image, or sync
# dependencies inside the container. Its /opt/plato environment is not mounted.
if [ "$#" -eq 0 ]; then
    set -- /bin/bash
fi
if [ -t 0 ] && [ -t 1 ]; then
    set -- -it plato "$@"
else
    set -- -i plato "$@"
fi
exec docker run --rm --net=host -v /dev/shm:/dev/shm \
    -v "$PWD:/root/plato" "$@"
