# AMLA: an AutoML frAmework for Neural Networks

AMLA is a framework for implementing and deploying AutoML algorithms for Neural Networks.

## Introduction

AMLA is a common framework to run different AutoML algorithms for neural networks without changing 
the underlying systems needed to configure, train and evaluate the generated networks. This has two benefits:
* It ensures that different AutoML algorithms can be easily compared
using the same set of hyperparameters and infrastructure, allowing for 
easy evaluation, comparison and ablation studies of AutoML algorithms.
* It provides a easy way to deploy AutoML algorithms on multi-cloud infrastructure.

With a framework, we can manage the lifecycle of autoML easily. Without this, hyperparameters and architecture design are spread out, some embedded in the code, others in config files and other as command line parameters, making it hard to compare two algorithms or perform ablation studies.

Some design principles of AMLA:
* The network generation process is decoupled from the training/evaluation process.
* The network specification model is independent of the implementation of the training/evaluation/generation code and ML library (i.e. whether it uses TensorFlow/PyTorch etc.).

AMLA currently supports the [NAC using EnvelopeNets](http://arxiv.org/pdf/1803.06744) AutoML algorithm, and we are actively adding newer algorithms to the framework. More information on AutoML algorithms for Neural Networks can be found [here](https://github.com/hibayesian/awesome-automl-papers)

## Architectural overview

In AMLA, an AutoML algorithm is run as a task and is specified through a configuration file. 
Sample configuration files may be found [here](./configs) and are described [here](./docs/config)

When run in single host mode (the default), the system consists of 
* Command Line Interface (CLI): An interface to add/start/stop tasks.
* Scheduler: Starts and stops the AutoML tasks.
* Generate/Train/Evaluate: The subtasks that comprise the AutoML task: network generation (via an AutoML algorithm), training and evaluation.

A more detailed description of the current architecture is available [here](./docs/design/arch.md)

The current branch is limited to operation on a single host i.e. the CLI, scheduler, generation, training and evaluation all run on a single host. The scheduler may be run as a service or a library, while the generate/train and evaluate subtasks are run as processes. A distributed system that allows concurrent execution of multiple training/evaluation tasks and distributed training on a pod of machines is under development.

## Contributing

At this point, AMLA is in its early stages. 
There are several areas in which development is yet to start or that are under development.
If you would like to contribute to AMLA's development, please send in pull requests, feature requests or submit proposals.
[Here](./CONTRIBUTING.md) is how to contribute.

Here are some areas that we need help with:
* [New AutoML algorithms]() Add support for new AutoML algorithms such as NAS, ENAS, AmoebaNet
* [Machine learning frameworks]() Add support for more machine learning frameworks such as PyTorch etc.
* [Standard model format](./docs/proposals/model/proposal.md) Improve the model specification (add hyper parameters), support for ONNX, other standard model specification formats.
* [Deployers](./docs/proposals/deployer/proposal.md): Add support for Kubeflow as a deployer 
* [Front end](./docs/proposals/fe/proposal.md): Contribute to ongoing development using vue.js
* [Test scripts](): Regression scripts, please!
* [Documentation](./docs): Documentation, please!

Proposals in progress are [here](./docs/proposals)
## Installation

### Prerequisites: 

Current AMLA supports Tensorflow as the default machine learning library. 
To install Tensorflow, follow the instructions here: 
- https://www.tensorflow.org/install/install_linux#InstallingVirtualenv
to install TensorFlow in a virtualenv for GPU/CPU.
- Alternatively, use an AWS DeepLearning AMI on an AWS GPU instance:
http://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/

### Install
```
    git clone https://github.com/ciscoai/amla
    cd amla/amla
    pip install -r requirements.txt
```

### Run the CLI
```
    python amla.py
```

### Add/start a task
```
Run an AutoML algorithm (NAC) to generate/train/evaluate 

#amla add_task configs/config.nac.construction.json
Added task: {'taskid': 0, 'state': 'init', 'config': 'configs/config.nac.construction.json'} to schedule.
#amla start_task 0

Start a single train/evaluate run using a network defined in the config file

#amla add_task configs/config.run.json
Added task: {'taskid': 1, 'state': 'init', 'config': 'configs/config.run.json'} to schedule.
#amla start_task <taskid>

Run the test construction algorithm (few iterations, few training steps)

#amla add_task configs/config.nac.construction.test.json
Added task: {'taskid': 2, 'state': 'init', 'config': 'configs/config.nac.construction.test.json'} to schedule.
#amla start_task <taskid> 

Run the test training/evaluation task

#amla add_task configs/config.run.test.json
Added task: {'taskid': 3, 'state': 'init', 'config': 'configs/config.run.test.json'} to schedule.
#amla start_task <taskid> 
```

Note: If the task run fails, kill the scheduler process,
remove the results/ directory, and restart amla

### Analyze
```
    tensorboard --logdir=amla/results/<arch name>/results/
```

## Running AutoML/NAS algorithms

Below are the configuration files to construct networks and run the constructed network for a few widely known AutoML/NAS algorithms.
In construction mode, AMLA generates networks using the algorithm.
In final network mode, AMLA runs the final network generated by an algorithm.
For some algorithms, we have included the config files for the construction as well as the final network.
For other algorithms, implementation is in progress, so we have provided only the final network 
configuration file, based on the network described in their papers/open source code.


| Algorithm             | Dataset            | Mode                     | Config file                      | AMLA result| Paper result    |
|-----------------------|--------------------|--------------------------|-------------------------------------------------------|----------------|-------------------|
| NAC/EnvelopeNets<sup>4</sup>      | CIFAR10            | Construction             |  [configs/config.nac.construction.json](amla/configs/config.nac.construction.json)    | 0.25 days| 0.25 days|
| NAC/EnvelopeNets      | CIFAR10            | Final network            |  [configs/config.nac.final.json](amla/configs/config.nac.final.json)           | 3.33%    | 3.33% |
| NAC/EnvelopeNets      | Imagenet           | Final network            |  [configs/config.nac.imgnet.json](amla/configs/config.nac.imgnet.json)           | 11.77%  | 15.36% |
| ENAS (Macrosearch)<sup>1</sup>   | CIFAR10            | Final network            |  [configs/config.enas.json](amla/configs/config.enas.json)   | 4.3%  | 4.23% |
| ENAS                  | CIFAR10            | Final network            |  [configs/config.enas-micro.json](amla/configs/config.enas-micro.json)   | In Progress | 2.89%  |
| AmoebaNet-B<sup>2</sup>           | CIFAR10            | Final network            |  [configs/config.amoebanet.b.json](amla/configs/config.amoebanet.b.json)| 5.32%  | 2.13% |
| DARTS<sup>3</sup>                | CIFAR10            | Final network            |  [configs/config.darts.json](amla/configs/config.darts.json)      | 4.22%  | 2.94% |

The results columns should be interpreted based on the mode (construction or final network).
In construction mode the result is the time to generate the network on a NVidia V100 GPU. 
In final network mode the result is the classification error rate.

To run an algorithm/network, run the command: 
```
amla add_task <config file>
```
where the config file is specified in the table above.


Note: 
* The Imagenet network for NAC/EnvelopeNets was based on the CIFAR10 construction hyperparameters modified to values commonly used for Imagenet.
* Tuning of some networks is in progress, so AMLA results may not yet match paper results.

## References

[1] "Efficient Neural Architecture Search via Parameter Sharing",
Hieu Pham, Melody Y. Guan, Barret Zoph, Quoc V. Le, Jeff Dean, https://arxiv.org/abs/1802.03268

[2] "Regularized Evolution for Image Classifier Architecture Search",
Esteban Real, Alok Aggarwal, Yanping Huang, Quoc V Le, https://arxiv.org/abs/1802.01548

[3] "DARTS: Differentiable Architecture Search",
Hanxiao Liu, Karen Simonyan, Yiming Yang, https://arxiv.org/abs/1806.09055

[4] "Neural Architecture Construction using EnvelopeNets",
Purushotham Kamath, Abhishek Singh, Debo Dutta, https://arxiv.org/abs/1803.06744


## Questions?

* Documentation: amla.readthedocs.org
* Twitter: @amla_ai
* Slack: ciscoai.slack.com/amla

## Authors
* Utham Kamath pukamath@cisco.com
* Abhishek Singh abhishs8@cisco.com
* Debo Dutta dedutta@cisco.com

If you use AMLA for your research, please cite this [paper](./docs/design/amla.pdf)

```
@INPROCEEDINGS{kamath18,
  AUTHOR = {P. Kamath and A. Singh and D. Dutta},
  TITLE = {{AMLA: An AutoML frAmework for Neural Network Design}}
  BOOKTITLE = {AutoML Workshop at ICML 2018},
  CITY = {Stockholm},
  MONTH = {July},
  YEAR = {2018},
  PAGES = {},
  URL = {}
}
```
