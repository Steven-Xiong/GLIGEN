# BSUB -o ./bjob_logs/test_4.29_txt_imageconcat_obj_txtgrounding_grad_trainwithcoco_testoncoco.%J

# BSUB -q gpu-compute

# BSUB -gpu "num=1:mode=shared:gmodel=NVIDIAA40" 
# BSUB -J GLIGEN_test

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/GLIGEN
bash infer.sh