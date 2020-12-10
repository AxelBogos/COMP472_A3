# :beers: COMP 472 – Assignment 3 :tiger:

[Repo URL](https://github.com/AxelBogos/COMP472_A3) <br>
---

Axel Bogos - 40077502 <br>
Luan Meiyi - 40047658 <br>
Xavier Morin - 40077865

---

## Preliminary Information

#### Libraries Used:
* pandas
* numpy
---

## How To Run 

---

Execute the main() function of ```main.py```to execute  ```GNB.runGNB()``` and ```LSTM.runLSTM()``` functions from their respective directories and save their results in the ```./results``` directory. Overall, this how the project is organized: 

These files are organized in a directory structure as follows: 
```
./root
│ main.py
| model.bin
│ __init__.py    
│ README.md
|   
└───data
|   │   covid_test_public.tsv
│   |   covid_training.tsv
└───GNB
│   │ NaiveBayesClassifier.py
│   │ runGNB.py
└───LSTM
|   |runLSTM.py
|   └───src 
│   │   │model.py
│   │   │util.py
└───results
│   │eval_lstm.txt
│   │eval_NB-BOW-FV.txt
│   │eval_NB-BOW-OV.txt
│   │trace_LSTM.txt
│   │trace_NB-BOW-FV.txt
│   │trace_NB-BOW-OV.txt

```

---
