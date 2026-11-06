#!/bin/sh
# Build the Overleaf upload package (sources + .bbl) from paper/; run after a successful local compile.
set -e
cd "$(dirname "$0")/../paper"
rm -f ../overleaf_labeler_eab.zip
zip -q -r ../overleaf_labeler_eab.zip main.tex main.bbl refs.bib sections tables figs
echo "wrote overleaf_labeler_eab.zip"
