# Team 3

Team 3 consists of Hazel, Jose and Eero


The task is to predict the sales of Rossmann stores 6 weeks ahead.
But is it a daily sale, or are we better of with a weekly sales?
For planning purposes the Rossmann administration does probably not care about daily sales.  They need rather to plan stock and logistics, which might be better done on a weekly level.

### Data
Data is the Rossmann data from Kaggle, which contains daily sales figures for 1115 stores in the period 2011 to 2015. 

### Our Model
In order to solve the Rossmann challege 4 different models have been used after the data analysis,cleaning and feature engineering. These have been evaluated using  teh required metrics (root mean squared percentage error.

-Linear Regression:  (simplest one). Not complex enough to get high score.

-RandomForest Linear Regression: Good score.

-Gradientboosting: highest.

-Prophet(time series): poor performance, many time independent variables.



## Getting started
```bash
# Create environment for running the model in: 
=======
'''bash
: Create environment: 

Conda create --name "Team3" python=3.8
# Use environment
Conda activate Team3
'''

## In a folder of your choice
# create a subdirectory and copy the materials needed
git clone https://github.com/eeroolli/RossmannSalesTeam3.git

cd RossmannSalesTeam3

# The data needs to be unpacked
# for the training and validation data
python data.py  

#  at test time run to use the test data instead
python data.py --test 1  
```



## Help
For more help and information, please consult the read readme_competition.md
=======
run
git clone https://github.com/eeroolli/RossmannSalesTeam3.git



