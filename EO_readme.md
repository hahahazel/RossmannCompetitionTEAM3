# Team 3

Team 3 consists of Hazel, Jose and Eero

The task is to predict the sales of Rossmann stores 6 weeks ahead.
But is it a daily sale, or are we better of with a weekly sales?
For planning purposes the Rossmann administration does probably not care about daily sales.  They need rather to plan stock and logistics, which might be better done on a weekly level.

### Data
Data is the Rossmann data from Kaggle, which contains daily sales figures for 1115 stores in the period 2011 to 2015. 

### Our Model
Describe our model




## Getting started
```bash
# Create environment for running the model in: 
conda env create -f environment.yml
# The environment does need more than python and we therefore have also run
# conda env export > environment.yml    # this creates a file with all conda dependencies.
# we have also requirements.txt that contains all pip dependencies. This file is created with
# pipreqs. 

# Use environment
conda activate Team3

## In a folder of your choice
# create a subdirectory and copy the materials needed
git clone git@github.com:eeroolli/RossmannCompetitionTEAM3.git

cd RossmannCompetitionTEAM3.git

# install packages that are not installed by conda
pip install --upgrade pip
pip install -r requirements.txt

# The data needs to be unpacked
# for the training and validation data
python data.py  

#  at test time run to use the test data instead
python data.py --test 1  



```

## Model



## Help
For more help and information, please consult the read readme_competition.md


