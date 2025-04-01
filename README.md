# Search to Integrate Multi-level Heuristics with Graph Neural Networks
for Multi-relational Link Prediction
---
## Requirements
```text
python=3.8
torch==1.9.0+cu111
dgl+cu111==0.6.1
```

## Instructions to run the experiment

### Search process
```shell
# Random baseline
python main.py --train_mode search --search_mode random --encoder SPATune --max_epoch 200

# supernet training
python main.py --train_mode search --search_mode spos --encoder SPASPOSSearch --search_max_epoch 1000

# architecture search
python main.py --train_mode search --search_mode spos_search --encoder SPASPOSSearch --arch_sample_num 1000 --weight_path <xx.pt>
```
### Fine-tuning process
```shell
python main.py --train_mode tune --encoder SPATune --search_res_file <xxx.json>
```
## Acknowledgement
The codes of this paper are partially based on the codes of [SANE](https://github.com/AutoML-Research/SANE) and [SPA](https://github.com/striderdu/SPA). We thank the authors of above work.
