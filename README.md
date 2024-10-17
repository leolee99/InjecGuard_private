<div align=center>

<h1>NotInject: Evaluating and Mitigating Over-defense in Prompt Guard Models for Robustness against Prompt Injection Attacks</h1>

<p align="center" style="overflow:hidden;">
 <img src="assets/3D.png" width="60%" style="margin: -2% -10% -0% -10%;">
</p>

</div>

## Requirements
We recommend the following dependencies.

* Python 3.10
* [PyTorch](http://pytorch.org/) 2.4.0

Then, please install other environment dependencies through:
```bash
pip install -r requirements.txt
```


## ⚙️ Dataset Preparation


We have provided our training dataset in the path of ```InjecGuard\datasets```.


## 🔥 How to Train

You can train InjecGuard by excuting the command:
```
python multi_task_train.py
```

## 📋 Evaluation

You can evalaute  trained InjecGuard by excuting the command:
```
python eval.py
```

