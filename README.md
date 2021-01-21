### Description of files:

`modelSearch.py` is how the models are tested and optimized and ensembled and saved.

`testSubsetSize.py` trains the selected super learner model on incremental proportions of training data to see the performance approach a maximum.

`modelExpand.py` takes every combination of possible missing values in an input patient and fits and saves a model to that specific combination. Prepares the expanded models for production use (saving models/LABEL/all_scores_IDENTIFIER.csv for use in figure generation) . Also, this allows for use in production to check for missing values in the input patient data and choose the correct model to predict accordingly.

`utils/cleaning.py` is how the raw data is cleaned to then be fit.

`utils/SuperLearner.py` is the super learner framework that I built. It holds the SuperLearner class that can be fit, output scores, and make predictions...

`utils/__init__.py` holds calibrateMeta, a function to wrap the super learner's meta model in a CalibratedClassifier. However, because the production model appears to be calibrated (see the calibration curve), it is not currently used in the study.

#### Sample slurm script (example `modelSearch.py` call) to find best models:

```Bash
#!/bin/bash
#SBATCH --job-name=study50kvit # create a short name for your job
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
module purge
module load anaconda3
conda activate visualize
l="study50k_allmodels_vit"
mkdir ./models/$l
python3 modelSearch.py -c=f -nccs=1000 -ss=50000 -rs=24 -v=t -sav=t -l=$l -ad=f
```

#### Example `modelExpand.py` call:
```Bash
python3 modelExpand.py -l="study50k_allmodels_vit" -o='OP' -rs=24 -p=0.85
```