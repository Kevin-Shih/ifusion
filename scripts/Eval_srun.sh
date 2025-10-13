#!/bin/bash
# Check for help flag
if [ "$#" -lt 2 ] || [ "$1" == "-h" -o "$1" == "-H" -o "$1" == "--help" ]; then
    echo "Usage: "$0" <Wandb_ID> <NVS_ROOT>"
    exit 1
fi
#==========================
# Confirm Basic Settings
#==========================
Wandb_ID="$1"
NVS_ROOT="$2"

echo "Wandb_ID="$'\t'"$Wandb_ID"
echo "NVS_ROOT="$'\t'"$NVS_ROOT"
read -p "Above setting correct? (Y/N)" -n 1 -r
echo 
echo 
if [[ $REPLY =~ ^[Yy]$ ]]
then
    #==========================
    # Load modules
    #==========================
    module reload
    module list
    echo 

    #==========================
    # Miniconda Environment
    #==========================

    eval "$(conda shell.bash hook)"

    conda deactivate
    conda activate ifusion

    #==========================
    # Execute Training
    #==========================
    srun -J "Eval_$Wandb_ID" -p defq --time=1:00:00 --tasks=1 --cpus-per-task=2 --gres=gpu:1 --pty \
        apptainer exec --nv -B /home:/home -B /cm:/cm -W $PWD ../apptainer-tools/ub24.sif python eval.py -g 0 -3 \
            --config config/main_scene_my_general.yaml \
            log.run_path="$Wandb_ID" \
            data.nvs_root="$NVS_ROOT"
fi