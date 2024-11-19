python scripts/owl_vit_eval.py \
 --model gpt-4 \
 --run_base_path img_generations/img_generations_templatev0.1_sd_lmd_gpt-4/run0 \
 --skip_first_prompts 0 \
 --prompt_start_ind 0 \
 --verbose \
 --detection_score_threshold 0.15 \
 --nms_threshold 0.15 \
 --class-aware-nms
