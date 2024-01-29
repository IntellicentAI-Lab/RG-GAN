## RG-GAN: Dynamic Regenerative Pruning for Data-Efficient Generative Adversarial Networks, AAAI 2024


### Abstract

---
Training Generative Adversarial Networks (GAN) to generate high-quality images typically requires large datasets. Network pruning during training has recently emerged as a significant advancement for data-efficient GAN. However, simple and straightforward pruning can lead to the risk of losing key information, resulting in suboptimal results due to GAN's competitive dynamics between generator (G) and discriminator (D). Addressing this, we present RG-GAN, a novel approach that marks the first incorporation of dynamic weight regeneration and pruning in GAN training to improve the quality of the generated samples, even with limited data. Specifically, RG-GAN initiates layer-wise dynamic pruning by removing less important weights to the quality of the generated images. While pruning enhances efficiency, excessive sparsity within layers can pose a risk of model collapse. To mitigate this issue, RG-GAN applies a dynamic regeneration method to reintroduce specific weights when they become important, ensuring a balance between sparsity and image quality. Though effective, the sparse network achieved through this process might eliminate some weights important to the combined G and D performance, a crucial aspect for achieving stable and effective GAN training. RG-GAN addresses this loss of weights by integrating learned sparse network weights back into the dense network at the previous stage during a follow-up regeneration step. Our results consistently demonstrate RG-GAN's robust performance across a variety of scenarios, including different GAN architectures, datasets, and degrees of data scarcity, reinforcing its value as a generic training methodology. Results also show that data augmentation exhibits improved performance in conjunction with RG-GAN. Furthermore, RG-GAN can achieve fewer parameters without compromising, and even enhancing, the quality of the generated samples.

### Impressive results

---
![Mian Figure](./figures/main_figure.jpg "Main Figure")

### Prerequisites

---
Our codes were implemented by Pytorch, we list the libraries and their version used in our experiments, but other versions should also be worked.
1. Linux         (Ubuntu)
2. Python        (3.8.0)
3. Pytorch         (1.13.0+cu116)
4. torchvision         (0.14.0)
5. numpy (1.23.4)

## Getting Started

---
### Usage
Should you have any questions about this repo, feel free to contact Jiahao Xu @ jiahaoxu@nevada.unr.edu


#### Hyperparameters introduction for SNGAN/RG-SNGAN

| Argument        | Type       | Description                                                               |
|-----------------|------------|---------------------------------------------------------------------------|
| `epoch`         | int        | Number of total training epochs                                           |
| `batch_size`    | int        | Batch size of per iteration, choose a proper value by yourselves          |
| `diffaug`         | store_true | Enable Diffaug training or not                                              |
| `lambda`             | float        | The penalty term                                                       |
| `data_ratio`    | float      | To simulate a training data limited scenario                              |



#### Data Preparation
Pytorch will download the CIFAR-10 dataset automatically if the dataset is not detected, therefore there is no need to prepare CIFAR-10 dataset.


#### Example

To run a RG-SNGAN model, you may follow:
1. Clone this repo to your local environment.
```
git clone https://github.com/IntellicentAI-Lab/RG-GAN.git
```
2. Prepare all the required libraries and datasets.


3. Run your model! One example can be:
```
# For SNGAN
python sngan_main.py --epoch 1000 --data_ratio 0.1 --eva_epoch 5 --diffaug 

# For RG-SNGAN
python rggan_main.py --epoch 1000 --data_ratio 0.1 --eva_epoch 5  --diffaug --lambda 1e-13
```

### Citation

___
If you use this code for your research, please cite our papers.


### Acknowledgment

___
We would like to thank the work that helps our paper:

1. FID score: https://github.com/bioinf-jku/TTUR.
2. Inception score: https://github.com/w86763777/pytorch-gan-metrics.
3. DiffAugmentation: https://github.com/VITA-Group/Ultra-Data-Efficient-GAN-Training.
4. SNGAN: https://github.com/w86763777/pytorch-gan-collections.


