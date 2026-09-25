datasets=("RAWFC" "LIAR-RAW")
models=("openai/gpt-4o-mini" "google/gemini-2.5-flash" "meta-llama/llama-3.3-70b-instruct")

for dataset in ${datasets[@]}; do
    for model_url in ${models[@]}; do
        uv run llm_classification_baseline.py --dataset $dataset --model_url $model_url
    done
done


