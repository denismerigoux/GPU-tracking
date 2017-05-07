#!/bin/bash

set -e

rm -rf build/*
CFLAGS=-D__STDC_CONSTANT_MACROS ./configure  --enable-avresample \
    --disable-debug \
    --disable-stripping \
    --enable-shared \
    --enable-version3
make
make DESTDIR=build install
cp build/usr/local/* /afs/andrew.cmu.edu/usr14/ideutel/builds-15618-project
