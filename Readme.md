## SeqGNN
### Introduction
The codes here include PyTorch implementations of the baseline and our SeqGNN model. Our code is based on [SGNN](https://github.com/eecrazy/ConstructingNEEG_IJCAI_2018). Code for EventComp model and how to extract the narrative event chains from raw NYT news corpus can be found [here](http://mark.granroth-wilding.co.uk/papers/what_happens_next/).

### Environmental dependence
+ Python 3.7.4
+ PyTorch 1.3.3
+ Red Hat 4.8.5-28
+ GPU (TITAN V)

### How to run the code?
First, you need to download the `data` and put it in the data folder. Data includes `deepwalk_128_unweighted_with_args.txt`.

Second, you can config the parameters of the model by `config.py` and run `main.py` to train the model.
```python
python main.py # train the seqgnn model
```
Third, you can run `chain.py` to train the baseline of SeqGNN-GRUFusion.
```python
python chain.py -l 4 -m train # train the 4th location
python chain.py -l 4 -m test # test the 4th location
```

Fourth, you can run `evaluate.py` to evaluate the accuracy of seqgnn model.
```python
python evaluate.py
```
