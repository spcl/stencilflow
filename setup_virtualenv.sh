#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)"
(cd $DIR && git submodule update --init --recursive)
(cd $DIR && virtualenv venv)
(cd $DIR && source venv/bin/activate)
(cd $DIR && pip install -r requirements.txt)
(cd $DIR && pip install --editable dace)
