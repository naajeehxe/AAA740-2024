CUDA_VISIBLE_DEVICES=2,3 python generate.py \
  --prompt-type lmd \
  --model StableBeluga2 \
  --save-suffix "gpt-4" \
  --repeats 1 \
  --regenerate 1 \
  --force_run_ind 0 \
  --run-model sd \
  --no-scale-boxes-default \
  --template_version v0.1