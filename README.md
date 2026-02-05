# Robust Learning with Time Series Noisy Labels via Self-supervised Learning and Soft Labels Refurbishment

This is the official code for our paper [*"Robust Learning with Time Series Noisy Labels via Self-supervised Learning and Soft Labels Refurbishment"*]().

## Abstract

Label noise significantly degrades the generalization performance of Deep Neural Networks (DNNs). While Learning with Noisy Labels (LNL) is well-established in computer vision, its application to time-series data presents unique challenges: 
(1) Feature extraction in supervised learning is corrupted by noisy labels, as the critical assumption that clean samples yield lower loss than noisy ones is frequently violated. This leads to learned representations with poor class separability and ambiguous boundaries, which degrades downstream task performance. 
(2) Conventional soft-labeling methods generate soft-labels from a single training epoch, ignoring historical and cross-model information. This often leads to unstable supervision and inconsistent soft-labels.
To address these challenges, this paper proposes the Robust Representation Learning Network (RoRLNet) for noisy time-series classification. RoRLNet employs a two-stage robust learning paradigm that decouples feature extraction from classifier training. In the first stage, it learns noise-robust spatio-temporal representations by integrating MixDecomposition, 
a data augmentation strategy based on trend-seasonality decomposition, with MSSFE, a multi-scale self-supervised feature extractor. In the second stage, it trains the classifier using EnBootstrap, a soft-label correction module that stabilizes supervision by ensembling predictions from multiple models and historical epochs. Extensive experiments on multiple benchmarks under diverse noise conditions demonstrate that RoRLNet consistently outperforms state-of-the-art methods, by 7.76\%.

## Datasets

### UEA 30 archive time series datasets

* [UEA 30 archive](http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip)

### Two individual large time series datasets

* [HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
* [ArabicDigits dataset]([https://github.com/imics-lab/TSAR](https://www.mustafabaydogan.com/research/time-series-data-mining/symbolic-representations-for-multivariate-time-series-classification-smts/))

## Usage

To train a RoRLNet model on a dataset, run

```bash
python main.py --archive UEA --dataset ArticularyWordRecognition --noise_type symmetric --label_noise_rate 0.2
```

## Citation

If you use this code for your research, please cite our paper:
