#!/bin/bash
# python eval.py -g 0 -n -c -3 -4 \
# apptainer exec --nv -B /home:/home -B /cm:/cm ../apptainer-tools/ub24.sif \
conda activate ifusion
python eval.py -g 0 -3 \
    --config config/main_scene_my_general.yaml \
    log.run_path="$1" \
    data.nvs_root="$2"