#!/bin/bash
patch -u -b ./plato/servers/base.py -i ./docs/patches/base.py.patch
patch -u -b ./plato/datasources/huggingface.py -i ./docs/patches/huggingface.py.patch
patch -u -b ./setup.py -i ./docs/patches/setup.py.patch
