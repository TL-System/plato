#!/bin/bash
patch -u -b ./plato/servers/base.py -i ./docs/patches/base.py.patch
patch -u -b ./plato/config.py -i ./docs/patches/config.py.patch
