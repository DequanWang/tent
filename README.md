# ⛺️ Tent: Fully Test-Time Adaptation by Entropy Minimization

This is the official project repository for [Tent: Fully-Test Time Adaptation by Entropy Minimization](https://openreview.net/forum?id=uXl3bZLkr3c) by
Dequan Wang\*, Evan Shelhamer\*, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell (ICLR 2021, spotlight).

Tent equips a model to adapt itself to new and different data ☀️ 🌧 ❄️  during testing.
Tent updates online and batch-by-batch to reduce error on dataset shifts like corruptions, simulation-to-real discrepancies, and other differences between training and testing data.

Our **example code** illustrates the method and provides representative results for image corruptions on CIFAR-10-C.
Note that the exact details of the model, optimization, etc. differ from the paper, so this is not for reproduction, but for explanation.

Please check back soon for our **reference code** to reproduce and extend tent!

## Example: Adapting to Image Corruptions on CIFAR-10-C

This example compares a baseline without adaptation (base), test-time normalization that updates feature statistics during testing (norm), and our method for entropy minimization during testing (tent).

- Dataset: [CIFAR-10-C](https://github.com/hendrycks/robustness/), with 15 corruption types and 5 levels.
- Model: [WRN-28-10](https://github.com/RobustBench/robustbench), the default model for RobustBench.

**Usage**:

```python
python cifar10c.py --cfg cfgs/base.yaml
python cifar10c.py --cfg cfgs/norm.yaml
python cifar10c.py --cfg cfgs/tent.yaml
```

**Result**: tent reduces the error (%) across corruption types at the most severe level of corruption (level 5).

|                                                      | mean | gauss_noise | shot_noise | impulse_noise | defocus_blur | glass_blur | motion_blur | zoom_blur | snow | frost |  fog | brightness | contrast | elastic_trans | pixelate | jpeg |
| ---------------------------------------------------- | ---: | ----------: | ---------: | ------------: | -----------: | ---------: | ----------: | --------: | ---: | ----: | ---: | ---------: | -------: | ------------: | -------: | ---: |
| [base](./cifar10c.py)       | 43.5 |        72.3 |       65.7 |          72.9 |         46.9 |       54.3 |        34.8 |      42.0 | 25.1 |  41.3 | 26.0 |        9.3 |     46.7 |          26.6 |     58.5 | 30.3 |
| [norm](./norm.py)       | 20.4 |        28.1 |       26.1 |          36.3 |         12.8 |       35.3 |        14.2 |      12.1 | 17.3 |  17.4 | 15.3 |        8.4 |     12.6 |          23.8 |     19.7 | 27.3 |
| [tent](./tent.py)       | 18.6 |        24.8 |       23.5 |          33.0 |         11.9 |       31.9 |        13.7 |      10.8 | 15.9 |  16.2 | 13.7 |        7.9 |     12.1 |          22.0 |     17.3 | 24.2 |

See the full results for this example in the [wandb report](https://wandb.ai/tent/cifar10c).

## Correspondence

Please contact Dequan Wang and Evan Shelhamer at dqwang AT cs.berkeley.edu and shelhamer AT google.com.

## Citation

If the tent method or fully test-time adaptation setting are helpful in your research, please consider citing our paper:

```bibtex
@inproceedings{
wang2021tent,
title={Tent: Fully Test-Time Adaptation by Entropy Minimization},
author={Dequan Wang and Evan Shelhamer and Shaoteng Liu and Bruno Olshausen and Trevor Darrell},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=uXl3bZLkr3c}
}
```
