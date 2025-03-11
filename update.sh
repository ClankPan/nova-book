#!/bin/bash
set -e  # stop script if error happens

pandoc README.md -s -o nova-book.html --mathjax