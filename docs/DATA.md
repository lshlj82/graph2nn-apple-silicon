# Setting Up Data Paths

For ImageNet, `tools/prepare_imagenet.py` builds this layout from either the Kaggle
download or the official image-net.org tarballs, and can make smaller subsets
(e.g. ImageNet-100) with symlinks. Run it with `-h` for examples.

Expected datasets structure for ImageNet:

```
imagenet
|_ train
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ val
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ ...
```

Expected datasets structure for CIFAR-10:

```
cifar10
|_ data_batch_1
|_ data_batch_2
|_ data_batch_3
|_ data_batch_4
|_ data_batch_5
|_ test_batch
|_ ...
```
