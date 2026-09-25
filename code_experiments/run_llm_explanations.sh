datasets=("RAWFC" "LIAR-RAW")
models=("meta-llama/llama-3.3-70b-instruct" "openai/gpt-4o-mini")

for dataset in ${datasets[@]}; do
    for model_url in ${models[@]}; do
        # uv run explanation_generation_deepinfra.py --dataset $dataset --model_url $model_url
        uv run explanation_generation.py --dataset $dataset --model_url $model_url
    done
done


