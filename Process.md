# Creating the environment

conda env create 

# Creating the requirements file
'''bash 
pip install pipreqs
pip install nbconvert
jupyter nbconvert --output-dir="./reqs" --to script *.ipynb
cd reqs
pipreqs ../requirements.txt

'''