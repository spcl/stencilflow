#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)"
(cd $DIR && git submodule update --init --recursive)
(cd $DIR && virtualenv venv)
source $DIR/venv/bin/activate
$DIR/venv/bin/pip install -r requirements.txt
$DIR/venv/bin/pip install --editable dace
