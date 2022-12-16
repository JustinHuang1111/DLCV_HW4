#!/bin/bash
python ./p2/evaluate.py --image_path $2 --csv_path $1 --output_path $3 --model_path1 ./p2/ckpt/400minesgd_first_finetune.pt --model_path2 ./p2/ckpt/300_fourth_finetune.pt

# TODO - run your inference Python3 code