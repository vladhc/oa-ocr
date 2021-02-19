#!/bin/bash

set -eu

shellcheck ./*.sh
echo "mypy checks.."
mypy ./*.py
echo "pycodestyle checks.."
pycodestyle ./*.py
echo "pylint checks.."
pylint ./**/*.py

coverage run --module unittest discover -p '*_test.py'
coverage report --show-missing --include="ocr/*,./train.py,./commons.py"
