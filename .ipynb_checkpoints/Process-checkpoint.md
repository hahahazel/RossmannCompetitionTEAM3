
# Creating the requirements file
'''bash 
pip install pipreqs
pip install nbconvert
jupyter nbconvert --output-dir="./reqs" --to script *.ipynb
cd reqs
pipreqs 
mv requirements.txt ..
'''