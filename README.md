# Parallelizing Model-based Reinforcement Learning Over the Sequence Length

Zirui Wang, Yue DENG, Junfeng Long, Yin Zhang

Paper & OpenReview: [Parallelizing Model-based Reinforcement Learning Over the Sequence Length](https://openreview.net/forum?id=R6N9AGyz13).

## Training and Evaluating Instructions

* Install the necessary dependencies. Note that we conducted our experiments using `python 3.8`.
  ```shell
  pip install -r requirements.txt
  ```

* Before Training, please set up [wandb](https://wandb.ai/site/) logger by changing
  ```
  os.environ["WANDB_API_KEY"] = [Your API Key]
  ```

* Then you can train the agent by running:
  ```
  # For Atari
  cd pwm_atari/
  python -u -O train.py \
    -n "[task]-life_done-pwm-100k" \
    -seed [seed] \
    -config_path [config path] \
    -env_name "ALE/[task]-v5" \
    -device [device]

  # For DMC Suite
  cd pwm_dmc/
  python -u -O train.py \
    -n "[task]-pwm" \
    -config_path [config path] \
    -obs_type ["visual" or "proprio"] \
    -env_name [task] \
    -device [device] \
    -seed [seed]
  ```

* The evaluation results will be presented in a CSV file located in the `eval_result` folder.
  ```
  # For Atari
  python -u -O eval.py \
    -env_name "ALE/[task]-v5" \
    -run_name "[task]-life_done-pwm-100k" \
    -config_path [config path] \
    -device [device] \
    -seed [seed]
  
  # For DMC Suite
  python -u -O eval.py \
    -n "[task]-["visual" or "proprio"]-pwm" \
    -config_path [config path] \
    -obs_type ["visual" or "proprio"] \
    -env_name [task] \
    -device [device] \
    -seed [seed]
  ```

## Reproducing Speed Metrics
To reproduce the speed metrics mentioned in the paper, please consider the following:
- Hardware requirements: NVIDIA GeForce RTX 3090/4090 with a high frequence CPU. Low frequence CPUs may slow down the traning.
- Software requiements: `PyTorch>=2.0.0` is required.


## Code references
We've referenced several other projects during the development of this code:
- [rmsnorm](https://github.com/bzhangGo/rmsnorm) For RMSNorm.
- [ChatRWKV](https://github.com/BlinkDL/ChatRWKV) For Token Mixing module.
- [STORM](https://github.com/BlinkDL/ChatRWKV) For training & evaluation configuration.
## Bibtex

```
@inproceedings{
    wangparallelizing,
    title={Parallelizing Model-based Reinforcement Learning Over the Sequence Length},
    author={Wang, ZiRui and Yue, DENG and Long, Junfeng and Zhang, Yin},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=R6N9AGyz13}
}
```