CUDA_VISIBLE_DEVICES=gpu_id PORT=29504 ./tools/dist_test.sh config_file_path \
    model_weight_path num_gpus \
    --eval 'MRE_i2','MRE_std_i2','SDR_2_i2','SDR_2.5_i2','SDR_3_i2','SDR_4_i2'