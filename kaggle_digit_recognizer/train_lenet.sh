#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/kaggle_digit_recognizer/lenet_solver.prototxt $@
