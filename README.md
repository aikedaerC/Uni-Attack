
## Summary
We evaluate 4 open source Vision-Language-Models(VLMs) performance (i.e. including mSSIM, Lingo-Judge, BLEURT, BERTScore and comming soon Attack Success Rate(ASR)) with 3 types of adversarial attacks methods (i.e. gradient-based transfer attack, typographic attack, and Both of them) on the task of decision making (i.e. specifically including color-judgement, object counting, object classification) in autonomous driving. 

## Schedule
So there is 3 steps overview. 
- Fistly, using the 3 types attacks methods to `generate attacked samples`. 
- Secondly, `getting the inference reuslts` on both original dataset and attacked dataset.
- Thirdly, `calculating the performance score` based on the inference resutls. 

## Step 0. Setting up Env

> 1. Attacking env 
- minigptv_environment.yml: for AttackVisionFoundationModels and AttackVLM

> 2. VLM env 
- imp_environment.yml: for Qwen-VL and imp-v1-3b
- llava_environment.yml: for LLaVA
- vila_environment.yml: for VILA 

> 3. Score env
- bleurt_environment.yml: for Bluert 
- lingoqa_environment.yml: for LingoJudge 
- bertscore_environment.yml: for BERTScore

## Step 1. Generate attacked samples

- Comparation with `AttackVLM`, `SGA`, `Co-attack`
- Ablation analysis on `noise`, `text`, and `noise_text`.

### AttackVLM 

The tgt images is limited to person.
```python
python _train_adv_img_trans_run.py
```

## Step 2. Get inference results

> Note. 
- question1 and question2 are the same, but for different datasets, question1 for phase1 and question2 for phase2. 
- For each image, question covers `object counting` and `color judgement`, without `classification`.
- The following scripts process VQA task, so `ground truth` and `prediction` are both generated by them.

```python
export CUDA_VISIBLE_DEVICES=3; python qwen_bbox.py 
export CUDA_VISIBLE_DEVICES=2; python vila.py      
export CUDA_VISIBLE_DEVICES=1; python imp_bbox.py  
export CUDA_VISIBLE_DEVICES=0; python llava.py     
```

## Step 3. Calculating performance scores

### mSSIM score

```python
python /home/anonymous/mydata/LVLMs/AttackVisionFoundationModels/ssim_cal_scripts.py batch path1 path2
```

### LingoJudge Score

```python
python /home/anonymous/mydata/LVLMs/LingoQA/eval_batch.py
```

### Bluert Score

Firstly, extract text from the predictions results

```
/home/anonymous/mydata/LVLMs/bleurt/ex.ipynb
```

```python
python /home/anonymous/mydata/LVLMs/bleurt/bleurtrun.py
python /home/anonymous/mydata/LVLMs/bleurt/avg.py
```

### BERTScore

```python
python /home/anonymous/mydata/LVLMs/bert_score/bert_score.py
```
