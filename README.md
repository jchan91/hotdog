# Notes
Has 2 steps.
1. Generating dataset (augmenting/normalizing images)
2. Training


## Generating dataset
Use generate_dataset.py. It will transform the input dataset into a training set. It will mirror the input directory.

Example:
```
python generate_dataset.py C:\data\hotdog C:\data\training\hotdog
```

The training step will assume that the folder name for images is the class label.


## Training
Use train.py on the output dataset from generate_dataset.py.


## Tools