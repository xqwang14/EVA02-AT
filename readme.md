# EVA02-AT: Egocentric Video-Language Understanding with Spatial-Temporal RoPEs and Symmetric Optimization

This is the offical implementation of [EVA02-AT Suite](https://arxiv.org/abs/2506.14356).

Xiaoqi Wang, Yi Wang, Lap-Pui Chau.

The Hong Kong Polytechnic University

## Setup

We use the same pytorch environment as AVION. For more details, please refer to [INSTALL.md](https://github.com/zhaoyue-zephyrus/AVION/blob/main/docs/INSTALL.md).

We also provide the docker environment. [BaiduNetDisk](https://pan.baidu.com/s/1MfDXywHN3Gj_buDMJYY2gA?pwd=1c5q)

In our docker environment, you can activate the conda environment by:

 ```commandline
    conda activate avion
 ```

 For data preparation, see [DATA.md](./DATA.md).


## Training and Evaluation

We provide a series of scripts in `./scripts` folder for training and evaluation.

* Pretrain the model on ego4d dataset:
     ```commandline
        sh pretrain_evab_egoclipplus.sh
    ```
The checkpoints of pretrained models are available here: [EVA02-AT-B](https://pan.baidu.com/s/1Upmz4GFT8AHsKTR-vBH9xw?pwd=4xcb) and [EVA02-AT-L](https://pan.baidu.com/s/17z1ELXqb0CyMhkaGhOFb8g?pwd=ks7n)

* Finetune the model for EK-100 MIR:
     ```commandline
        sh mir_ft_evab.sh
    ```
Please make sure the pretrained models are put into `./pretrain` first.

The checkpoints of finetuned models are avaible here: [EVA02-AT-B](https://pan.baidu.com/s/1yGR7gMFvILh9mfwWbhytYA?pwd=hb38) and [EVA02-AT-L](https://pan.baidu.com/s/1cYPahQCBV-0YCWw2-5V5vQ?pwd=rvns)

* Finetune the model on CharadesEgo dataset:
     ```commandline
        sh ft_charades.sh
    ```

* Evaluate the model on CharadesEgo dataset:
     ```commandline
        sh test_charades.sh
    ```

* Evaluate the model on EgoMCQ benchmark:
     ```commandline
        sh egomcq_evab.sh
    ```

## License

[MIT License](./LICENSE).

## Acknowledgements

This repository is built upon [AVION](https://github.com/zhaoyue-zephyrus/AVION). we thank the authors for releasing their code.

## Citation

```bibtex
@article{eva02_at,
      title={{EVA02-AT}: Egocentric Video-Language Understanding with Spatial-Temporal Rotary Positional Embeddings and Symmetric Optimization}, 
      author={Xiaoqi Wang and Yi Wang and Lap-Pui Chau},
      journal={arXiv preprint arXiv:2506.14356},
      year={2025}
}
```

```bibtex
@article{sms,
      title={Symmetric Multi-Similarity Loss for EPIC-KITCHENS-100 Multi-Instance Retrieval Challenge 2024}, 
      author={Xiaoqi Wang and Yi Wang and Lap-Pui Chau},
      journal={arXiv preprint arXiv:2406.12256},
      year={2024}
}
```