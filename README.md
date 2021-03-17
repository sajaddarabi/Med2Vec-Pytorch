# Med2Vec in Pytorch

This is a re-implementation of Med2Vec [1] in Pytorch. It simply embeds clinical concepts into a distributed representation using skip-gram model with an aditional code loss.

To run the code first obtain the `ADMISSION.CSV` and `DIAGNOSES_ICD.CSV` from MIMIC-III database [here](https://mimic.physionet.org/).

Compile the data by running `bash gen_data.sh` make sure you set the correct paths to the files.

The directories are structured as follows:
- `./base`: base trainer, data loader.
- `./configs`: json files for experiments. This where you pass in arguments to the model/trainer and what have you.
- `./trainer`: contains training logic, and anything that must be done to train the model.
- `./model`: directory containing the med2vec model.


To train the model run the following:

`python train_med2vec.py -c ./configs/config.json`

**note:** make sure the directories are set appropiatly in `./configs/config.json`. 

[1] Choi, Edward, et al. "Multi-layer representation learning for medical concepts." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016.
