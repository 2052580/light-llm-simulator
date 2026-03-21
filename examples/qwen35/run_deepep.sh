rm -rf ./output.log
cur_date=$(LC_ALL=en_US.utf8 date + "%b%d-%H%M") && echo $cur_date
export LOG_LEVEL=INFO
python examples/qwen35/deepep.py > data/output-${cur_date}.log 2>&1
ln -s data/output-${cur_date}.log ./output.log
echo "saved perf datas to data/output-${cur_date}.log"
python src/visualization/throughput.py \
--serving_mode "DeepEP" \
--model_type "Qwen/Qwen3.5-397B-A17B" \
--device_type "Ascend_A3Pod" \
--tpot_list 50 \
--kv_len_list 4096
