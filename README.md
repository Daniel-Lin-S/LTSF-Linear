# Research in progress ...

This repo is the develouped from the framework of time series prediction experiment of LTSF-Linear. (https://github.com/cure-lab/LTSF-Linear)


## Benchmark Models

Forecasting models included unde the framework:
- [x] [Transformer](https://arxiv.org/abs/1706.03762) (NeuIPS 2017)
- [x] [Informer](https://arxiv.org/abs/2012.07436) (AAAI 2021 Best paper)
- [x] [Autoformer](https://arxiv.org/abs/2106.13008) (NeuIPS 2021)
- [x] [Pyraformer](https://openreview.net/pdf?id=0EXmFzUn5I) (ICLR 2022 Oral)
- [x] [FEDformer](https://arxiv.org/abs/2201.12740) (ICML 2022)
- [x] [PatchTST](https://arxiv.org/abs/2211.14730) (ICLR 2023)
- [x] [Linears](https://arxiv.org/pdf/2205.13504.pdf) (AAAI-23 Oral)


## Detailed Description
Experiment script files in `./scripts`:
| Files      |                              Interpretation                          |
| ------------- | -------------------------------------------------------| 
| EXP-LongForecasting      | Long-term Time Series Forecasting Task                    |
| EXP-LookBackWindow      | Study the impact of different look-back window sizes   | 
| EXP-Embedding        | Study the effects of different embedding strategies (for transformers)   |
| EXP-STFT        | Study hyper-parameter sensitivity of STFT based models  |



## Getting Started
### Environment Requirements

You may want to create a virtual environment to hold all the dependencies of this project.

To use conda:
```
conda create -n $env_name python=3.12.7
conda activate $env_name
which pip  # check that the pip being used is the one in virtual environment
pip install -r requirements.txt
```

To use pip:
```
python3 -m venv .$env_name
source .$env_name/bin/activate
pip install -r requirements.txt
```

### Data Preparation

You can obtain all the nine benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. All the datasets are well pre-processed and can be used easily.

```
mkdir dataset
```
**Please put the csv files in the `./dataset` directory**

To use a custom dataset, it must be converted into csv file with a `date` column containing the time stamps.

The usual data loading is completed using `data_provider.data_loader.DatasetCustom`,
which uses train-validate-test ratio of 7:1:2.
The training data and test data are obtained by adjoint sliding windows
with hop length set default to 1. (You can modify this)

If you only need to extract data from a specified section of time:
- create a subclass of the class `data_provider.data_loader.BaseDataset`, and extract data using self-defined splitting boundaries. (see `Dataset_ETT_hour` as an example)
- add this to the file `dataprovider/data_factory.py`, giving a name to your Dataset
- change the argument `data` correspondingly when running test scripts


### Models and Training
All models are stored under the folder `model`, and the training, validation and testing
procedures are written under `exp`. 

- `exp_main.py`: the primary experiment used for end-to-end training of machine learning methods. It is used by `run_longExp.py` to train models.
  - Each of the model file used by `Exp_Main` should define a class `Model(nn.Module)` with `forward`, `train`, `test` methods implemented. (back-propagation and validation are completed by `Exp_Main`)
  - Early stopping is implemented and the paticence can be set by parser argument `patience`
  - The optimiser used is Adam, set learning rate by `learning_rate` argument.
  - See details of all tunable training arguments in `run_longExp.py`
- `exp_stat.py`: used for statistical models that require no training. (no trainable parameters)
  - All the statistical models can be found in `models/Stat_models.py`.

To add a new machine learning model: 
- modify the `model_dict` in `models/exp_main.py` and modify the method `get_pred_model_settings` in `utils/configs.py` to specify how the model should be named by the folders and csv files storing results.
  - e.g. Linear with `individual=True` would be named as `Linear_indTrue`.
- Remember to give the model attribute `requires_time_markers` indicating whether `forward` takes only `x` as input, or `(x, x_mark, decoder_input, y_mark)` as input where `x_mark, y_mark` are the time markers, typically used for transformer embeddings.

To use a model whose training is not supported by the existing experiments, you need to create a subclass of `Exp_Basic`, implementing fuctions like `train`, `vali`, `test`, `_get_data` (which dataset to use) and `_build_model` (model initialisation).

All the test resutls are saved in `results.csv` by default.


### Training Example
- In `scripts/ `, we provide the model implementation *Dlinear/Autoformer/Informer/Transformer*
- In `FEDformer/scripts/`, we provide the *FEDformer* implementation
- In `Pyraformer/scripts/`, we provide the *Pyraformer* implementation

For example:

To train the **LTSF-Linear** on **Exchange-Rate dataset**, you can use the script `scripts/EXP-LongForecasting/Linear/exchange_rate.sh`:
```
sh scripts/EXP-LongForecasting/Linear/exchange_rate.sh
```
It will start to train DLinear by default, the results will be shown in `logs/LongForecasting`. You can specify the name of the model in the script. (Linear, DLinear, NLinear)

All scripts about using LTSF-Linear on long forecasting task is in `scripts/EXP-LongForecasting/Linear/`, you can run them in a similar way. The default look-back window in scripts is 336, LTSF-Linear generally achieves better results with longer look-back window as dicussed in the paper. 

Scripts about look-back window size and long forecasting of FEDformer and Pyraformer are in `FEDformer/scripts` and `Pyraformer/scripts`, respectively. To run them, you need to first `cd FEDformer` or `cd Pyraformer`. Then, you can use sh to run them in a similar way. Logs will be stored in `logs/`.

Each experiment in `scripts/EXP-LongForecasting/Linear/` takes 5min-20min. For other Transformer scripts, since we put all related experiments in one script file, directly running them will take 8 hours per day. You can keep the experiments you are interested in and comment on the others. 


## Citing

If you find this repository useful for your work, please consider citing it as follows:

```BibTeX

```

Please remember to cite all the datasets and compared methods if you use them in your experiments.
