# Tested with 2 & 4 GPUs

set -x
# if [ "$#" -lt 2 ]; then
#     echo "Usage: run_qwen3_8b_sft_peft_sp2_npu.sh <nproc_per_node> <save_path> [other_configs...]"
#     exit 1
# fi

# nproc_per_node=$1
# save_path=$2

nproc_per_node=2
save_path=/data/czxu/xa/models/checkpoints/verl_sft_example_scad/qwen3_8b_sft_3120_highdata_shuffle_samepromt1g

# Shift the arguments so $@ refers to the rest
# data.train_files='["/home/xa/data/scad_verl_filter/train-00000-of-00008.parquet","/home/xa/data/scad_verl_filter/train-00001-of-00008.parquet","/home/xa/data/scad_verl_filter/train-00002-of-00008.parquet","/home/xa/data/scad_verl_filter/train-00003-of-00008.parquet","/home/xa/data/scad_verl_filter/train-00004-of-00008.parquet","/home/xa/data/scad_verl_filter/train-00005-of-00008.parquet","/home/xa/data/scad_verl_filter/train-00007-of-00008.parquet" ]' \
# data.val_files=/home/xa/data/scad_verl_filter/test_temp_100.parquet \
# data.train_files='["/data/czxu/xa/scad_verl_filter_sft/train-00000-of-00008.parquet","/data/czxu/xa/scad_verl_filter_sft/train-00001-of-00008.parquet","/data/czxu/xa/scad_verl_filter_sft/train-00002-of-00008.parquet","/data/czxu/xa/scad_verl_filter_sft/train-00003-of-00008.parquet","/data/czxu/xa/scad_verl_filter_sft/train-00004-of-00008.parquet","/data/czxu/xa/scad_verl_filter_sft/train-00005-of-00008.parquet","/data/czxu/xa/scad_verl_filter_sft/train-00007-of-00008.parquet" ]' \

shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    verl/trainer/fsdp_sft_trainer.py \
    data.train_files='["/data/czxu/xa/myc1g_verl_sft_toenf_sameprompt_3072_sf_newcot_high/train-00000-of-00008.parquet","/data/czxu/xa/myc1g_verl_sft_toenf_sameprompt_3072_sf_newcot_high/train-00001-of-00008.parquet","/data/czxu/xa/myc1g_verl_sft_toenf_sameprompt_3072_sf_newcot_high/train-00002-of-00008.parquet","/data/czxu/xa/myc1g_verl_sft_toenf_sameprompt_3072_sf_newcot_high/train-00003-of-00008.parquet","/data/czxu/xa/myc1g_verl_sft_toenf_sameprompt_3072_sf_newcot_high/train-00004-of-00008.parquet","/data/czxu/xa/myc1g_verl_sft_toenf_sameprompt_3072_sf_newcot_high/train-00005-of-00008.parquet","/data/czxu/xa/myc1g_verl_sft_toenf_sameprompt_3072_sf_newcot_high/train-00007-of-00008.parquet" ]' \
    data.val_files=/data/czxu/xa/myc1g_verl_sft_toenf_sameprompt_3072_sf_newcot_high/test_temp_150.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=5120 \
    data.truncation=right \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=/data/czxu/xa/models/qwens/Qwen3-8B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=scad-sft \
    trainer.experiment_name=qwen3_8b_sft_3120_highdata_shuffle_samepromt1g \
    trainer.logger=console \
    trainer.total_epochs=15 $@ \
    trainer.test_freq=20 \
    model.target_modules=all-linear \
    trainer.save_freq=42 \
    # Or you can do this:
    # model.target_modules=[q_proj,v_proj] \
