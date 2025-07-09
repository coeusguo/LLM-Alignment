python rft_ray.py \
-lora \
-training-step 1000 \
-eval-every 1 \
-normalize-by-std \
-batch-size 2 \
-num-samples-per-prompt 8 \
-accumulation-steps 4 \
-rollout-batch-size 8 \
-epochs-per-rollout-batch 2 \
-log-save-path ckpt \
-lr 1e-5 \
-weight-decay 0.0 \


#-model qwen2-0.5b \
#-ckpt-path /data/coding/assignment5-alignment/ckpts/qwen2-0.5b_gsm8k_SFT_step_1000 \