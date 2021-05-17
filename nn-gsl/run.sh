#!/bin/bash

clang nn.c -o nn `gsl-config --cflags` `gsl-config --libs` && ./nn