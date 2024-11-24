CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
  --prompt-type demo \
  --model gpt-4 \
  --save-suffix "gpt-4" \
  --repeats 1 \
  --frozen_step_ratio 0.5 \
  --regenerate 1 \
  --force_run_ind 1 \
  --run-model lmd_plus \
  --no-scale-boxes-default \
  --template_version v0.1 \
  # --verbose True
#   --force_run_ind 2