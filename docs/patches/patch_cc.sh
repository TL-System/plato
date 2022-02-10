#!/bin/bash
patch -u -b ./plato/servers/base.py -i ./docs/patches/base.py.patch
