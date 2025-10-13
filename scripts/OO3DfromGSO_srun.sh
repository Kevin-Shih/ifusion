#!/bin/bash
# Check for help flag
if [ "$1" == "-h" -o "$1" == "-H" -o "$1" == "--help" ]; then
    echo "Usage: "$0" [RUN_NAME] [NVS_ROOT] [RESUME_CKPT_FP]"
    exit 1
fi
#==========================
# Confirm Basic Settings
#==========================
RUN_NAME="OO3DfromGSO900_${1:-"test"}" #OO3D_FromGSO900_General_bs32_newlr6
NVS_ROOT="runs/Fixed_scene_pose/${2:-$RUN_NAME}"
RESUME_CKPT_FP="runs/Fixed_scene_pose/my_general_600step_accum8/GSO/lora/${3:-"lora_900.ckpt"}"

echo "RUN_NAME="$'\t'"$RUN_NAME"
echo "NVS_ROOT="$'\t'"$NVS_ROOT"
echo "RESUME_CKPT_FP="$'\t'"$RESUME_CKPT_FP"
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
    srun -J $RUN_NAME -p defq --time=4:00:00 --tasks=1 --cpus-per-task=8 --gres=gpu:1 --pty \
        python main.py -g 0 -6 \
            --config config/main_scene_my_general.yaml \
            log.run_name="$RUN_NAME" \
            data.nvs_root="$NVS_ROOT" \
            data.resume_ckpt_fp="$RESUME_CKPT_FP"
fi