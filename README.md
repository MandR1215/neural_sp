# Learning Neural Strategy-Proof Matching Mechanism from Examples
This repository officially implements ***Learning Neural Strategy-Proof Matching Mechanism from Examples***.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python3 check_train.py \
        --matcher <name_of_matching_algorithm> \
        --n_workers <number_of_workers> \
        --n_firms <number_of_firms> \
        --epochs <epochs> \
        --seed <seed> \
        --batch_size <batch_size> \
        --input_size <dimension_of_contexts>\
        --train_size <number_of_train_instances>
```

We used `--seed 42` for all the experiments.

## Evaluation

To evaluate your model compared to RSD, run:

```eval
python3 test.py\
        --matcher <name_of_matching_algorithm>\
        --n_workers <number_of_workers>\
        --n_firms <number_of_firms>\
        --seed <seed>\
        --input_size <dimension_of_contexts>\
        --test_size <number_of_test_instances>\
```

To evaluate your model on small scales and compute recovering rate, run the above by replacing `test.py` with `test_small.py`.
We used `--seed 1` for all the experiments, excluding the 'recovery rate' result with `--seed 1` to `--seed 20`.
