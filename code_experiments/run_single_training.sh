export MLFLOW_TRACKING_URI=http://127.0.0.1:7777
device=$1
class_name=$2
config_path=$3

# optional parameters to override config
plm_dropout=$4
graph_type=$5
finetuning_type=$6

uv run python training_general.py \
    --dataset "RAWFC"\
    --seed 42\
    --path_data "nli_relations_r10-s5_deberta-large"\
    --model_class $class_name\
    --model_config_path $config_path\
    --device $device
    # --plm_dropout_prob $plm_dropout\
    # --finetuning_type $finetuning_type\
    # --graph_type "heterogeneous"\