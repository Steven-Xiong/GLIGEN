# BSUB -o ./bjob_logs/train_5.1_text_dino_COCO_flickr_vg_sbu.%J

# BSUB -q gpu-compute
# BSUB -m "a100s-2305.engr.wustl.edu" 

# BSUB -gpu "num=4:mode=shared:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_80GB" 
# BSUB -J anydoor

source ~/.bashrc
conda activate instdiff
cd /project/osprey/scratch/x.zhexiao/GLIGEN
bash train.sh