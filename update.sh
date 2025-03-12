#!/bin/bash
set -e  # stop script if error happens

pandoc ./docs/README.md -o docs/index.html --mathjax  --toc --standalone