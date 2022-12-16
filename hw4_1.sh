#!/bin/bash
python ./p1/run.py --config configs/nerf/hotdog_test.py --render_test --render_only --dump_images --json_dir $1 --outpath $2
# TODO - run your inference Python3 code