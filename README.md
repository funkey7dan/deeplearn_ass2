# README
======
Daniel Bronfman, ***REMOVED***
Coral Kuta, 208649186

## Requirements:
-	Python 3.x
-	PyTorch
-	torchtext
-	numpy
-	matplotlib
-	argparse


## File structure:
```
├── images
├── ner
│   ├── dev
│   ├── test
│   └── train
├── pos
│   ├── dev
│   ├── test
│   └── train
├── tagger1.py
├── tagger2.py
├── tagger3.py
├── tagger4.py
├── tagger_launcher.py
├── top_k.py
├── tree.txt
├── vocab.txt
└── wordVectors.txt
```
Note: this directory structure is required for the plot functions to work

## Relevant files:
### Files for all questions:
-	`tagger_launcher.py`
-	`./ner/test		- not submitted
-	`./ner/train	- not submitted
-	`./ner/dev		- not submitted
-	`./pos/test		- not submitted
-	`./pos/train	- not submitted
-	`./pos/dev		- not submitted

### Files for question 1:
-	`tagger1.py`
-	`test1.pos`
-	`test1.ner`

### Files for question 2:
-	`top_k.py`

### Files for question 3:
-	`tagger2.py`
-	`test3.pos`
-	`test3.ner`

### Files for question 4:
-	`tagger3.py`
-	`test4.pos`
-	`test4.ner`

### Files for question 5:
-	`tagger4.py`
-	`test5.pos`
-	`test5.ner`


## Instructions:
1.	Make sure all the requirements above are fulfilled.
2. 	Download the relevant files.
3. 	Run tagger_launcher with the following arguments (in this order):
	python tagger_launcher.py [-h] [--part PART] [--task TASK] [--pretrained]
   
	The script accepts the following command-line arguments:

	--part, -p: Specifies which part of the exercise to run. It should be an integer between 1 and 5 (inclusive).
	--task, -t: Specifies the task to run, either "ner" or "pos".
	--pretrained, --pre: (Optional) Specifies whether to use pretrained embeddings or not. Only applicable for tagger3. If provided, the script will use pretrained embeddings.
	Note: All the arguments are optional, but at least one argument is required to execute the script.
	
## Examples:
Run part 1 for named entity recognition (NER):
python tagger_launcher.py --p 1 --t ner

Run part 2:
python tagger_launcher.py --part 2

Run part 4 for part-of-speech (POS) tagging without pretrained embeddings:
python tagger_launcher.py --p 4 --t pos

Run part 4 for NER with pretrained embeddings:
python tagger_launcher.py --p 4 --t ner --pre

Run part 5 for POS tagging: 
python tagger_launcher.py --part 5 --task pos
