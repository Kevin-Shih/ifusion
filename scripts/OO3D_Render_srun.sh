#!/bin/bash
# Check for help flag
if [ "$#" -lt 2 ] || [ "$1" == "-h" -o "$1" == "-H" -o "$1" == "--help" ]; then
    echo "Usage: "$0" <RUN_NAME> <LoRA_ITER> [NVS_ROOT]"
    exit 1
fi
#==========================
# Confirm Basic Settings
#==========================
RUN_NAME="Render${2}_OO3DfromGSO900_${1}" #OO3D_FromGSO900_General_bs32_newlr6
LoRA_CKPT_FP="runs/Fixed_scene_pose/OO3DfromGSO900_${1}/OO3D/lora/lora_${2}.ckpt"
NVS_ROOT="runs/Fixed_scene_pose/${3:-Render${2}_OO3DfromGSO900_${1}}"

echo "RUN_NAME="$'\t'"$RUN_NAME"
echo "LoRA_CKPT_FP="$'\t'"$LoRA_CKPT_FP"
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
    srun -J $RUN_NAME -p defq --time=40:00 --tasks=1 --gres=gpu:1 --pty \
        python main.py -g 0 -7 \
            --config config/main_scene_my_general.yaml \
            log.run_name="$RUN_NAME" \
            data.nvs_root="$NVS_ROOT" \
            data.lora_ckpt_fp="$LoRA_CKPT_FP"
fi