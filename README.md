# C3030

To reduce data without using jupyter notebooks, use reduction_setup.py which in turn uses loadmirdata.py and reduction.py

load data
python loadmirdata.py C '2016-04-09'
                       ^band    ^--- directory containing raw, blocks, reduced_band folders
python reduction.py C '2016-04-09'
 
when using Xgterm, there's some bug where the script hangs until you close Xgterm


With the notebooks, first load the data with loadmirdata.py, then you can use reduction_batchrun.ipynb, which runs mirpy_reduction_template.ipynb

--> you have to quit xgterm otherwise the notebook with hang indefinitely

For the model fitting, I use analysis_batchrun.ipynb, which runs C3030_modelling_template.ipynb

After model fitting, I read in the generated evidence values using population_analysis.ipynb

Miriad_tutorial follows the tutorial devolped by Jamie Stevens (http://www.atnf.csiro.au/computing/software/miriad/tutorials.html) but instead uses mirpy
