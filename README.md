# EVA02-AT Suite: Egocentric Video Models with Spatial-Temporal RoPEs

This is the offical implementation of EVA02-AT Suite.
Xiaoqi Wang, Yi Wang, Lap-Pui Chau.
The Hong Kong Polytechnic University

## Installation

We use the same pytorch environment as AVION. See [INSTALL.md](docs/INSTALL.md) to install this code.

We also provide the docker environment.
## Model Zoo


    a. Zero-Shot performance after pretrained on EgoClip+ dataset:

    | Method |    Backbone   | batch-size<br>per GPU | GPU memory | Hardware | GPU×hour^ | EK100 MIR<br>0-shot Avg. mAP |
    | :----: | :-----------: | :-------------------: | :--------: | :------: | :-------: | :--------------------------: |
    | AVION  |      ViT-B    |          32           |     25     | 32× V100 |   1824    |            30.9              |
    |  Ours  |   EVA02-AT-B  |         256           |     19     | 8× A5000 |    260    |            33.2              |

## License

[MIT License](./LICENSE).

## Acknowledgements

