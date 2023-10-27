This is the code for our paper ```Counterfactual Representation Learning```.

The repository contains three folders. The ```data``` directory stored the information about the two datasets we used in the experiments. ```CVAE``` and ```DCEVAE``` include the code for the experiments in CVAE and DCEVAE causal graphs respectively.

To obtains the results shown in table 1, you should run

```sh
python3 main.py --use_label True  --run 1
```
for the ```UF```, ```DCEVAE```, ```CR``` baselines. 

For the ```I-DCEVAE``` baseline, you should run
```sh
python3 main.py --use_label True --use_real True  --run 1
```

For the ```CE``` baseline and our method, the comman is
```sh
python3 main.py --a_f 0.0 --u_kl 0.5 --a_h 0.1  --run 1
```

The results in table 2 are obtained by
```sh
python3 main.py --use_label True  --run 1
python3 main.py --use_label True --use_real True  --run 1
python3 main.py  --run 1
```

The results in table 3 are obtained by
```sh
python3 main.py --dataset law --a_y 1 --a_r 1 --a_d 1 --a_f 0.2 --a_h 0.4 --u_kl 1 --n_epochs 2000 --lr 1e-3 --use_label True --normalize True --run 1
python3 main.py --dataset law --a_y 1 --a_r 1 --a_d 1 --a_f 0.2 --a_h 0.4 --u_kl 1 --n_epochs 2000 --lr 1e-3 --use_label True --use_real True --normalize True --run 1
python3 main.py --dataset law --a_y 1 --a_r 1 --a_d 1 --a_f 0.2 --a_h 0.4 --u_kl 1 --n_epochs 2000 --lr 1e-3 --normalize True --run 1
```

The results in table 4 are obtained by
```sh
python3 main.py --dataset law --a_y 1 --a_r 1 --a_d 1 --a_f 0.15 --u_kl 1  --n_epochs 2000 --lr 1e-3 --use_label True --normalize True --run 1
python3 main.py --dataset law --a_y 1 --a_r 1 --a_d 1 --a_f 0.15 --u_kl 1  --n_epochs 2000 --lr 1e-3 --use_label True --normalize True --use_real True --run 1
python3 main.py --dataset law --a_y 1 --a_r 1 --a_d 1 --a_f 0.15 --u_kl 1 --n_epochs 2000 --lr 1e-3 --normalize True --run 1
```

For the experiments with path dependent counterfactual fairness, you need to add ```--path True``` and ```--path_attribute attr``` in the command. For the law school dataset, attr should be ```GPA``` or ```SAT```. And for the UCI dataset, attr should be ```0``` for ```workingclass``` and ```1``` for ```education```.

******************************************************************************
The detailed experiment results are stored in ```CF Data Record.xlsx```. You can also find the comman (including the random seed) used to get the results.

To get the Figure 5 and 9 shown in the paper, run
```sh
python3 draw.py
```