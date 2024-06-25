export TOOLBENCH_KEY=MhuKsGSC8inUrAjJ2ae0uymRJTnx0rZ8wtIb5tXqgpmmpDrjou
export PYTHONPATH=./

python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model toolllama \
    --model_path ToolBench/ToolLLaMA-7b \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method DFS_woFilter_w2 \
    --input_query_file data/instruction/inference_query_demo.json \
    --output_answer_file data/answer/toolllama_dfs \
    --toolbench_key $TOOLBENCH_KEY


#    --backbone_model toolllama \
python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model orca3 \
    --model_path /home/wchen/repos/orca3/orca3_ckpt_2000 \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method DFS_woFilter_w2 \
    --input_query_file data/instruction/inference_query_demo.json \
    --output_answer_file data/answer/toolllama_dfs \
    --toolbench_key $TOOLBENCH_KEY

python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model toolllama \
    --model_path /home/wchen/repos/orca3/orca3_v2_epoch_2 \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method DFS_woFilter_w2 \
    --input_query_file data/instruction/inference_query_demo.json \
    --output_answer_file data/answer/toolllama_dfs \
    --toolbench_key $TOOLBENCH_KEY


