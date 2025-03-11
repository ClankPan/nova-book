#!/bin/bash
set -e  # stop script if error happens

pandoc ./docs/README.md -s -o docs/index.html --mathjax --css ./style.css
# pandoc ./docs/README.md -s -o docs/index.html --mathjax -H ./docs/style.html