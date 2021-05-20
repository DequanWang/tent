# ⛺️ Tent: Fully Test-Time Adaptation by Entropy Minimization

This is the official project repository for [Tent: Fully-Test Time Adaptation by Entropy Minimization](https://openreview.net/forum?id=uXl3bZLkr3c) by
Dequan Wang\*, Evan Shelhamer\*, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell (ICLR 2021, spotlight).

⛺️ Tent equips a model to adapt itself to new and different data during testing ☀️ 🌧 ❄️.
Tented models adapt online and batch-by-batch to reduce error on dataset shifts like corruptions, simulation-to-real discrepancies, and other differences between training and testing data.
This kind of adaptation is effective and efficient: tent makes just one update per batch to not interrupt inference.

We provide **[example code](#example-adapting-to-image-corruptions-on-cifar-10-c)** in PyTorch to illustrate the tent method and fully test-time adaptation setting.

Please check back soon for **reference code** to exactly reproduce the ImageNet-C results in the paper.

**Installation**:

```
pip install -r requirements.txt
```

tent depends on

- Python 3
- [PyTorch](https://pytorch.org/) >= 1.0

and the example depends on

- [RobustBench](https://github.com/RobustBench/robustbench) v0.1 for the dataset and pre-trained model
- [yacs](https://github.com/rbgirshick/yacs) for experiment configuration

but feel free to try your own data and model too!

**Usage**:

```
import tent

model = TODO_model()

model = tent.configure_model(model)
params, param_names = tent.collect_params(model)
optimizer = TODO_optimizer(params, lr=1e-3)
tented_model = tent.Tent(model, optimizer)

outputs = tented_model(inputs)  # now it infers and adapts!
```

## Example: Adapting to Image Corruptions on CIFAR-10-C

The example adapts a CIFAR-10 classifier to image corruptions on CIFAR-10-C.
The purpose of the example is explanation, not reproduction: exact details of the model architecture, optimization settings, etc. may differ from the paper.
That said, the results should be representative, so do give it a try and experiment!

This example compares a baseline without adaptation (source), test-time normalization for updating feature statistics during testing (norm), and our method for entropy minimization during testing (tent).
The dataset is [CIFAR-10-C](https://github.com/hendrycks/robustness/), with 15 types and 5 levels of corruption.

### WRN-28-10

the default model for [RobustBench](https://github.com/RobustBench/robustbench).

**Usage**:

```python
python cifar10c.py --cfg cfgs/source.yaml
python cifar10c.py --cfg cfgs/norm.yaml
python cifar10c.py --cfg cfgs/tent.yaml
```

**Result**: tent reduces the error (%) across corruption types at the most severe level of corruption (level 5).

|                                                            | mean | gauss_noise | shot_noise | impulse_noise | defocus_blur | glass_blur | motion_blur | zoom_blur | snow | frost |  fog | brightness | contrast | elastic_trans | pixelate | jpeg |
| ---------------------------------------------------------- | ---: | ----------: | ---------: | ------------: | -----------: | ---------: | ----------: | --------: | ---: | ----: | ---: | ---------: | -------: | ------------: | -------: | ---: |
| source [code](./cifar10c.py)   [config](./cfgs/source.yaml)       | 43.5 |        72.3 |       65.7 |          72.9 |         46.9 |       54.3 |        34.8 |      42.0 | 25.1 |  41.3 | 26.0 |        9.3 |     46.7 |          26.6 |     58.5 | 30.3 |
| norm   [code](./norm.py)       [config](./cfgs/norm.yaml)         | 20.4 |        28.1 |       26.1 |          36.3 |         12.8 |       35.3 |        14.2 |      12.1 | 17.3 |  17.4 | 15.3 |        8.4 |     12.6 |          23.8 |     19.7 | 27.3 |
| tent   [code](./tent.py)       [config](./cfgs/tent.yaml)         | 18.6 |        24.8 |       23.5 |          33.0 |         12.0 |       31.8 |        13.7 |      10.8 | 15.9 |  16.2 | 13.7 |        7.9 |     12.1 |          22.0 |     17.3 | 24.2 |

See the full results for this example in the [wandb report](https://wandb.ai/tent/cifar10c/reports/Tent-Example-Image-Corruptions--Vmlldzo1NTA0NzM).

### WRN-40-2

WideResNet for [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781).

**Usage**:

```python
python cifar10c.py --cfg cfgs/source.yaml MODEL.ARCH Hendrycks2020AugMix_WRN
python cifar10c.py --cfg cfgs/norm.yaml MODEL.ARCH Hendrycks2020AugMix_WRN
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Hendrycks2020AugMix_WRN
```

**Result**: tent reduces the error (%) across corruption types at the most severe level of corruption (level 5).

|                                                            | mean | gauss_noise | shot_noise | impulse_noise | defocus_blur | glass_blur | motion_blur | zoom_blur | snow | frost |  fog | brightness | contrast | elastic_trans | pixelate | jpeg |
|------------|-----:|------------:|-----------:|--------------:|-------------:|-----------:|------------:|----------:|-----:|------:|-----:|-----------:|---------:|--------------:|---------:|-----:|
| source     | 18.3 |        28.8 |       23.0 |          26.2 |          9.5 |       20.6 |        10.6 |       9.3 | 14.2 |  15.3 | 17.5 |        7.6 |     20.9 |          14.7 |     41.3 | 14.7 |
| norm       | 14.5 |        18.5 |       16.2 |          22.3 |          9.0 |       21.9 |        10.5 |       9.7 | 12.8 |  13.3 | 15.0 |        7.6 |     11.9 |          16.3 |     15.0 | 17.5 |
| tent       | 12.1 |        15.7 |       13.2 |          18.8 |          7.9 |       18.1 |         9.0 |       8.0 | 10.4 |  10.8 | 12.4 |        6.7 |     10.0 |          14.0 |     11.4 | 14.8 |

## Example: Adapting to Adversarial Perturbations on CIFAR-10
See [Fighting Gradients with Gradients: Dynamic Defenses against Adversarial Attacks](https://arxiv.org/abs/2105.08714) for more details on [dent](https://github.com/DequanWang/dent).

## Correspondence

Please contact Dequan Wang and Evan Shelhamer at dqwang AT cs.berkeley.edu and shelhamer AT google.com.

## Citation

If the tent method or fully test-time adaptation setting are helpful in your research, please consider citing our paper:

```bibtex
@inproceedings{wang2021tent,
  title={Tent: Fully Test-Time Adaptation by Entropy Minimization},
  author={Wang, Dequan and Shelhamer, Evan and Liu, Shaoteng and Olshausen, Bruno and Darrell, Trevor},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=uXl3bZLkr3c}
}
```
