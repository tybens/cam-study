### Description of files:

`modelSearch.py` is how the models are tested and optimized and ensembled and saved.

`cleaning.py` is how the raw data is cleaned to then be fit.

`prepModel.py` shows how I convert the metamodel to a CalibratedClassifier as well as implement class weights and 'partially fit' it on all of the patient data n=200,000. It also shows how the content of `production_data/` is made and saved. 

`modelExpand.py` takes every combination of possible missing values in an input patient and fits and saves a model to that specific combination. This then uses prepModel.py to prepare the expanded models for production use (saving models/LABEL/all_scores_IDENTIFIER.csv for use in figure generation) . Also, this allows `productionApp.py` to check for missing values in the input patient data and choose the correct model to predict accordingly.


#### Example `prepModel.py` call:
```Bash
python3 prepModel.py -b=3 -v=f -l='cw_100k1kcc_nvit' -o='OB' -lo=f
```

#### Example `modelExpand.py` call (after `prepModel.py` is called):
```Bash
python3 modelExpand.py -l='cw_100k1kcc_nvit'
```

#### Sample slurm script (example `modelSearch.py` call) to find best models:

```Bash
#!/bin/bash
#SBATCH --job-name=mprocw100k1kcc_vit	 # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=7        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=50G         # memory per cpu-core (4G is default)
#SBATCH --time=200:00:00          # maximum time needed (HH:MM:SS)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail 	 # send email if job fails
#SBATCH --mail-user=tb19@princeton.edu
echo "||||||||||||||||||||||||||||||||||||||||||||||||||"
echo "--------------------------------------------------"
echo "Now with class weights and PRC AUC scoring!      "
module purge
module load anaconda3
conda activate visualize
l="mpro_cw_100k1kcc_vit"
mkdir ./models/$l
python3 modelSearch.py -c=f -nccs=1000 -ss=100000 -rs=51 -b=3 -v=t -sav=t -l=$l -ad=f
```