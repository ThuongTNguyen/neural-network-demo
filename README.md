A demo for neural network models to represent water dimer potential energy surface. 
=========================
This contains an example code to demonstrate the procedure to build neural network models for [the paper](https://aip.scitation.org/doi/10.1063/1.5024577) in The Journal of Chemical Physics by Thuong Nguyen et. al. 

Note that the public code here is not in the final optimal version used either for software development or for publication. It also has shorter training iterations than the actual version. 

## Directory content
* [`neuralnet-demo.ipynb`](https://github.com/ThuongTNguyen/neural-network-demo/blob/master/neuralnet-demo.ipynb): main training notebook - intro, data exploratory, feature engineering, model building/training/evaluation.
* [`src_nogrd.py`](https://github.com/ThuongTNguyen/neural-network-demo/blob/master/src_nogrd.py): a module containing source codes for all functions used in the main training notebook.
* `*.png`: supporting plots.

## Requirements
- Python 3.6
- Keras 2.0.8
- Numpy 1.12
- Pandas 0.19
- GPU (here Nvidia Tesla K40 or GeForce GTX 1080) with CUDNN v6

## License
This repo contains the code developed by Thuong Nguyen initially for the work published [here](https://aip.scitation.org/doi/10.1063/1.5024577) under the following license: 

  Copyright (c) 2018, Thuong T. Nguyen, Andreas W. Goetz, Andrea Zonca, Francesco Paesani
