#!/bin/bash
#SBATCH --account=def-lockedh
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100:1
#SBATCH --mem=32000M

echo This is job $SLURM_JOBID on $(hostname) at $(date)
echo SLURM_TMPDIR: $SLURM_TMPDIR
echo HOME: $HOME
echo SCRATCH: $SCRATCH
echo

#interactive session
#pip freeze > requirements.txt #saves specific versions, can be loaded
#pip install --no-index -r requirements.txt

module purge
module load python/3.12.4
module load gcc opencv scipy-stack cuda
module list #check if the below are installed
#virtualenv --no-download ENV
#source ENV/bin/activate
#pip install --no-index --upgrade pip
#pip install --no-index accelerate==1.13.0 rasterio==1.4.3 albumentations==2.0.0 tqdm==4.67.3 timm==1.0.24 #gotta get the right wheels

#stage files:
cp -r $HOME/sem_annotator/ $SLURM_TMPDIR/
exec &> $SLURM_TMPDIR/session.out
source $SLURM_TMPDIR/sem_annotator/ENV/bin/activate
cd $SLURM_TMPDIR/sem_annotator

echo Beginning yolo_to_mask.py at $(date)...
python yolo_to_mask.py --image-dir ./data/images --label-dir ./data/labels --output-dir ./data/masks

echo Beginning Tile_generator.py at $(date)...
python Tile_generator.py --img_dir ./data/images/ --mask_dir ./data/masks --tile_size 1024 --no-gui

cd train
echo Beginning train_comparison.py at $(date)...
accelerate launch train_comparison.py \
  --method canny \
  --image-dir ../data/images/tiles_output/images/ \
  --mask-dir ../data/images/tiles_output/masks/ \
  --coverage 0.5 \
  --target-size 1024 \
  --epochs 1000 \
  --batch-size 32 \
  --output-base ./SEM-SAM \
  --val-split 0.1 \
  --test-split 0.1 \
  --patience 100 \
  --mixed-precision bf16

cp -r $SLURM_TMPDIR/sem_annotator/train/SEM-SAM/ $SCRATCH/runs/test_run_001/
cp $SLURM_TMPDIR/session.out $SCRATCH/logs/

echo Done at $(date)
