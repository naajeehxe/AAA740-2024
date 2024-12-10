# 코드 흐름.

*_generate.py_* 

*lmd_plus.py_*
    def (run)
    so box 준비
    so prompt encoding 해서 준비
    
*generate_single_object_with_box_*




먼저 주어진 (spec) 을 convert


Spec을 converting 한 후에는 다음과 같이 주어진다.

**so_prompt_phrase_word_box_list**:

[('An indoor scene with a blue cube', 'a blue cube', 'cube', (0.39453125, 0.234375, 0.609375, 0.44921875)), 

('An indoor scene with a red cube', 'a red cube', 'cube', (0.39453125, 0.44921875, 0.609375, 0.6640625)), 

('An indoor scene with a vase', 'a vase', 'vase', (0.12109375, 0.37109375, 0.27734375, 0.6640625))]

**overall_prompt** : An indoor scene with a blue cube, a red cube, a vase

**overall_phrases_words_bboxes**: List.

[('a blue cube', 'cube', [(0.39453125, 0.234375, 0.609375, 0.44921875)]), 

('a red cube', 'cube', [(0.39453125, 0.44921875, 0.609375, 0.6640625)]), 

('a vase', 'vase', [(0.12109375, 0.37109375, 0.27734375, 0.6640625)])]


이후 lmd_plus.py

    so_boxes = [item[-1] for item in so_prompt_phrase_word_box_list]

에서 single object box를 얻고

*latents.get_input_latents_list* 로 들어감

latent 처리 후 

lmd_plus.py 의 get_masked_latents_all_list로 들어감.

