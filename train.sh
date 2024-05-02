# python main.py --name=COCO  --yaml_file=/project/osprey/scratch/x.zhexiao/GLIGEN/configs/coco2017K.yaml
python -m torch.distributed.launch --nproc_per_node=4 main.py \
--name=text_dino_ground_coco+flickr+vg+sbu  \
--yaml_file=/project/osprey/scratch/x.zhexiao/GLIGEN/configs/coco_text_dino.yaml \
--ckpt /project/osprey/scratch/x.zhexiao/GLIGEN/gligen_checkpoints/checkpoint_generation_text_image.bin \
--batch_size=16

# -m torch.distributed.launch --nproc_per_node=2