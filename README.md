# EMPIRICAL EVALUATIONS OF PERSONALIZED FEDERATED LEARNING ON HETEROGENEOUS ELECTRONIC HEALTH RECORDS

The original code of federated learning algorithms is available [here](https://github.com/TsingZ0/PFLlib).

## Environments
Install [CUDA](https://developer.nvidia.com/cuda-11-6-0-download-archive). 

Install [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) and activate conda. 

```bash
conda env create -f env_cuda_latest.yaml # You may need to downgrade the torch using pip to match CUDA version
```


## Data
MIMIC-IV-ED data used can be obtained by first accessing official website [here](https://physionet.org/content/mimiciv/2.2/) and then following the processing pipelines in this [paper](https://www.nature.com/articles/s41597-022-01782-9).

SGH-ED data is not available to the public due to third-party restrictions.

## Usage
For FL and PFL algorithms, run:
```
python main.py -datp /Users/dataset_path/ -data dataet_name -m lr -algo PerAvg -gr 3 -did 0 -nb 2 -nc 2 -le 1,1 -lr 1 -lbs 128
```
- `-datp`: path of the dataset.
- `-data`: name of the dataset.
- `-m`: name of the model. eg: -lr means logistics regression.
- `-algo`: federated learning algorithm(FedAvg, PerAvg, pFedMe).
- `-nb`: number of classes for classification problem.
- `-nc`: number of clients.
- `-le`: local epochs. eg: -le 1,10 means 1 local epochs for client1, 10 local epochs for client2.
- `-gr`: global rounds.
- `-lr`: learning rate.
- `-lbs`: local batch size.

For local models, run:
```
Local.ipynb
```

For parallel ablation studies, run:
```
python3 commands.py
```

## References
Johnson, Alistair EW, et al. "MIMIC-IV, a freely accessible electronic health record dataset." *Scientific data* 10.1 (2023): 1.

Fallah, Alireza, Aryan Mokhtari, and Asuman Ozdaglar. "Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach." Advances in Neural Information Processing Systems 33 (2020): 3557-3568.
