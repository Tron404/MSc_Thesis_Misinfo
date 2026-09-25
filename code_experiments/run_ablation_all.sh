export MLFLOW_TRACKING_URI=http://127.0.0.1:7777

dataset="RAWFC"

#### TEXT ONLY
uv run python training_general.py \
    --dataset $dataset\
    --path_data "nli_relations_r10-s5_deberta-large"\
    --seed 42\
    --model_class "TextOnly"\
    --model_config_path "model_configs/ablation_text_only.yaml"\
    --device "cuda:0" \
    --finetuning_type "none" &

uv run python training_general.py \
    --dataset $dataset\
    --path_data "nli_relations_r10-s5_deberta-large"\
    --seed 42\
    --model_class "TextOnly"\
    --model_config_path "model_configs/ablation_text_only.yaml"\
    --device "cuda:1" \
    --finetuning_type "lora"

#### GNN ONLY

## NO FINETUNING
uv run python training_general.py \
    --dataset $dataset\
    --path_data "nli_relations_r10-s5_deberta-large"\
    --seed 42\
    --model_class "GnnOnlyMain"\
    --model_config_path "model_configs/ablation_gnn_only.yaml"\
    --device "cuda:0" \
    --finetuning_type "none" \
    --graph_type "homogeneous" &


uv run python training_general.py \
    --dataset $dataset\
    --path_data "nli_relations_r10-s5_deberta-large"\
    --seed 42\
    --model_class "GnnOnlyMain"\
    --model_config_path "model_configs/ablation_gnn_only.yaml"\
    --device "cuda:1" \
    --finetuning_type "none" \
    --graph_type "heterogeneous" 

## LORA
uv run python training_general.py \
    --dataset $dataset\
    --path_data "nli_relations_r10-s5_deberta-large"\
    --seed 42\
    --model_class "GnnOnlyMain"\
    --model_config_path "model_configs/ablation_gnn_only.yaml"\
    --device "cuda:0" \
    --finetuning_type "lora" \
    --graph_type "homogeneous" &


uv run python training_general.py \
    --dataset $dataset\
    --path_data "nli_relations_r10-s5_deberta-large"\
    --seed 42\
    --model_class "GnnOnlyMain"\
    --model_config_path "model_configs/ablation_gnn_only.yaml"\
    --device "cuda:1" \
    --finetuning_type "lora" \
    --graph_type "heterogeneous" 

#### MLP
uv run python training_general.py \
    --dataset $dataset\
    --path_data "nli_relations_r10-s5_deberta-large"\
    --seed 42\
    --model_class "Mlp"\
    --model_config_path "model_configs/ablation_mlp.yaml"\
    --device "cuda:0" \
    --finetuning_type "none" &


uv run python training_general.py \
    --dataset $dataset\
    --path_data "nli_relations_r10-s5_deberta-large"\
    --seed 42\
    --model_class "Mlp"\
    --model_config_path "model_configs/ablation_mlp.yaml"\
    --device "cuda:1" \
    --finetuning_type "lora" \