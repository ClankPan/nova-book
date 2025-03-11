#!/bin/bash
set -e  # stop script if error happens

pandoc README.md -s -o docs/index.html --mathjax --css docs/style.css