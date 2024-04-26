import torch 
import json 
from collections import defaultdict
from PIL import Image, ImageDraw
from copy import deepcopy
import os 
from io import BytesIO
from zipfile import ZipFile 
import multiprocessing


from tsv import TSVFile, TSVWriter

from io import BytesIO
import base64
from PIL import Image
import numpy as np
import time 
from tqdm import tqdm

from pycocotools.coco import COCO
# ============= Useful fuctions and classes from Haotian =============== #
######################### COOL STUFF: NEEDED!!!! #############################


def encode_pillow_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')

def encode_tensor_as_string(arr):
    if type(arr) != np.ndarray:
        arr = arr.data.cpu().numpy()
    return base64.b64encode(arr.tobytes()).decode('utf-8')


def tensor_to_base64(image_tensor):
    """
    Converts a PyTorch tensor to a base64 encoded image.
    :param image_tensor: A PyTorch tensor of shape [C, H, W] with data range [0, 1].
    :return: A base64 encoded string of the image.
    """
    # Assume the tensor is in the range [0, 1] and has shape [C, H, W]
    # Convert it to PIL Image
    image_tensor = image_tensor.detach()  # Detach tensor from any computation graph
    image_tensor = image_tensor.mul(255).byte()  # Scale to [0, 255] and convert to byte
    image_tensor = image_tensor.cpu().numpy()  # Move tensor to CPU and convert to numpy array
    if image_tensor.shape[0] == 1:  # If grayscale
        image_pil = Image.fromarray(image_tensor.squeeze(0), mode='L')
    else:  # Assume RGB
        # Convert tensor order from [C, H, W] to [H, W, C]
        image_pil = Image.fromarray(image_tensor.transpose(1, 2, 0))

    # Encode PIL image to base64
    buffer = BytesIO()
    image_pil.save(buffer, format="JPEG")
    encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return encoded_string

def item_to_encodable(item):
    # print(item['image'].size)
    # import pdb; pdb.set_trace()
    item['image'] = encode_pillow_to_base64(item['image'])
    
    for anno in item['annos']:
        anno['text_embedding_before'] = encode_tensor_as_string(anno['text_embedding_before'])
        anno['image_embedding_before'] = encode_tensor_as_string(anno['image_embedding_before'])
        anno['text_embedding_after'] = encode_tensor_as_string(anno['text_embedding_after'])
        anno['image_embedding_after'] = encode_tensor_as_string(anno['image_embedding_after'])
        # print(anno['ref_mask'].shape)
        # anno['ref_mask'] = tensor_to_base64(anno['ref_mask'][0])
        # import pdb; pdb.set_trace()
        anno['ref_mask'] = tensor_to_base64(anno['ref_mask'].unsqueeze(0))
        anno['ref_img'] = tensor_to_base64(anno['ref_img'])
        anno['ref_box'] = encode_tensor_as_string(anno['ref_box'])
        
    return item


######################### COOL STUFF: NEEDED!!!! #############################
# ============= Useful fuctions and classes from Haotian =============== #





def check_unique(images, fields):
    for field in fields:
        temp_list = []
        for img_info in images:
            temp_list.append(img_info[field])
        assert len(set(temp_list)) == len(temp_list), field

def clean_data(data):
    for data_info in data:
        data_info.pop("original_img_id", None)
        data_info.pop("original_id", None)
        data_info.pop("sentence_id", None)  # sentence id for each image (multiple sentences for one image)
        data_info.pop("dataset_name", None)  
        data_info.pop("data_source", None) 
        data_info["data_id"] = data_info.pop("id")


def clean_annotations(annotations):
    for anno_info in annotations:
        anno_info.pop("iscrowd", None) # I have checked that all 0 for flickr, vg, coco
        #anno_info.pop("category_id", None)  # I have checked that all 1 for flickr vg. This is not always 1 for coco, but I do not think we need this annotation
        anno_info.pop("area", None)
        # anno_info.pop("id", None)
        anno_info["data_id"] = anno_info.pop("image_id")


def draw_box(img, boxes):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline ="red", width=2) # x0 y0 x1 y1 
    return img 


def xyhw2xyxy(box):
    x0, y0, w, h = box
    return [ x0, y0, x0+w, y0+h ]


class Base():
    def __init__(self, image_root):
        self.image_root = image_root
        self.use_zip = True if image_root[-4:] == ".zip" else False 
        if self.use_zip:
            self.zip_dict = {}

    def fetch_zipfile(self, ziproot):
        pid = multiprocessing.current_process().pid # get pid of this process.
        if pid not in self.zip_dict:
            self.zip_dict[pid] = ZipFile(ziproot)
        zip_file = self.zip_dict[pid]
        return zip_file

    def fetch_image(self, file_name):
        if self.use_zip:
            zip_file = self.fetch_zipfile(self.image_root)
            image = Image.open( BytesIO(zip_file.read(file_name)) ).convert('RGB')
        else:
            image = Image.open(  os.path.join(self.image_root,file_name)   ).convert('RGB')
        return image



class GroundingDataset(Base):
    "This is for grounding data such as GoldG, SBU, CC3M, LAION"
    def __init__(self, image_root, json_path, annotation_embedding_path):
        super().__init__(image_root)
        self.image_root = image_root
        self.json_path = json_path
        self.annotation_embedding_path = annotation_embedding_path

        # Load raw data 
        with open(json_path, 'r') as f:
            json_raw = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
        self.data = json_raw["images"] # donot name it images, which is misleading
        self.annotations = json_raw["annotations"]
      
        # clean data and annotation
        check_unique( self.data, ['id'] )
        check_unique( self.annotations, ['id'] )
        clean_data(self.data)
        clean_annotations(self.annotations)
        self.data_id_list = [  datum['data_id'] for datum in self.data   ]
        self.data = { datum['data_id']:datum  for datum in self.data } # map self.data from a list into a dict 

        # data point to its annotation mapping 
        self.data_id_to_annos = defaultdict(list)
        for anno in self.annotations:
            self.data_id_to_annos[ anno["data_id"] ].append(anno)

    def __getitem__(self, index):

        out = {}

        data_id = self.data_id_list[index]
        out['data_id'] = data_id
        
        file_name = self.data[data_id]['file_name']
        image = self.fetch_image(file_name)
        out["image"] = image
        out["file_name"] = file_name
        # import pdb; pdb.set_trace()
        out["caption"] = self.data[data_id]["caption"]

        annos = deepcopy(self.data_id_to_annos[data_id])
        
        for anno in annos:
            anno["text_embedding_before"] = torch.load(os.path.join(self.annotation_embedding_path,"text_features_before",str(anno["id"])), map_location='cpu') 

            anno["image_embedding_before"] = torch.load(os.path.join(self.annotation_embedding_path,"image_features_before",str(anno["id"])), map_location='cpu') 
            
            anno["text_embedding_after"] = torch.load(os.path.join(self.annotation_embedding_path,"text_features_after",str(anno["id"])), map_location='cpu') 
            
            anno["image_embedding_after"] = torch.load(os.path.join(self.annotation_embedding_path,"image_features_after",str(anno["id"])), map_location='cpu') 
            
            

        out["annos"] = annos

        return out

    def __len__(self):
        return len(self.data_id_list)


# class COCODataset(Base):
#     "This is for grounding data such as GoldG, SBU, CC3M, LAION"
#     def __init__(self, image_root, json_path, annotation_embedding_path):
#         super().__init__(image_root)
#         self.image_root = image_root
#         self.json_path = json_path
#         self.annotation_embedding_path = annotation_embedding_path

#         # Load raw data 
#         with open(json_path, 'r') as f:
#             json_raw = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
#         self.data = json_raw["images"] # donot name it images, which is misleading
#         self.annotations = json_raw["annotations"]
      
#         # clean data and annotation
#         check_unique( self.data, ['id'] )
#         check_unique( self.annotations, ['id'] )
#         clean_data(self.data)
#         clean_annotations(self.annotations)
#         self.data_id_list = [  datum['data_id'] for datum in self.data   ]
#         self.data = { datum['data_id']:datum  for datum in self.data } # map self.data from a list into a dict 

#         # data point to its annotation mapping 
#         self.data_id_to_annos = defaultdict(list)
#         for anno in self.annotations:
#             self.data_id_to_annos[ anno["data_id"] ].append(anno)

#     def getitem(self, index):

#         out = {}

#         data_id = self.data_id_list[index]
#         out['data_id'] = data_id
        
#         file_name = self.data[data_id]['file_name']
#         image = self.fetch_image(file_name)
#         out["image"] = image
#         out["file_name"] = file_name
#         import pdb; pdb.set_trace()
#         out["caption"] = self.data[data_id]["caption"]

#         annos = deepcopy(self.data_id_to_annos[data_id])
        
#         for anno in annos:
#             anno["text_embedding_before"] = torch.load(os.path.join(self.annotation_embedding_path,"text_features_before",str(anno["id"])), map_location='cpu') 

#             anno["image_embedding_before"] = torch.load(os.path.join(self.annotation_embedding_path,"image_features_before",str(anno["id"])), map_location='cpu') 
            
#             anno["text_embedding_after"] = torch.load(os.path.join(self.annotation_embedding_path,"text_features_after",str(anno["id"])), map_location='cpu') 
            
#             anno["image_embedding_after"] = torch.load(os.path.join(self.annotation_embedding_path,"image_features_after",str(anno["id"])), map_location='cpu') 
            
#             # add ref mask:
#             anno['ref_mask'] = torch.load( os.path.join(self.annotation_embedding_path,"ref_mask",str(anno["id"])), map_location='cpu'  )

#         out["annos"] = annos

#         return out

#     def __len__(self):
#         return len(self.data_id_list)

class COCODataset(Base):
    "This only supports instance_json, thus only for O365"
    def __init__(self, image_root, instances_json_path, annotation_embedding_path):
        super().__init__(image_root)

        self.image_root = image_root
        self.instances_json_path = instances_json_path
        self.annotation_embedding_path = annotation_embedding_path
        self.captions_json_path = '/project/osprey/scratch/x.zhexiao/GLIGEN/data/coco/annotations/captions_val2017.json'
        self.coco_caps = COCO(self.captions_json_path)
        # Load all jsons 
        with open(instances_json_path, 'r') as f:
            instances_data = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
        # self.annotations = instances_data["annotations"]
        clean_annotations(instances_data["annotations"])
        self.instances_data = instances_data

        # Misc  
        self.image_ids = [] # main list for selecting images
        self.image_id_to_filename = {} # file names used to read image
        for image_data in self.instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename

        
        # All category names (including things and stuff)
        self.object_idx_to_name = {} 
        for category_data in self.instances_data['categories']:
            self.object_idx_to_name[category_data['id']] = category_data['name']


        # Add object data from instances and stuff 
        self.image_id_to_objects = defaultdict(list)
        for object_anno in self.instances_data['annotations']:
            image_id = object_anno['data_id']
            self.image_id_to_objects[image_id].append(object_anno)
        

    def getitem(self, index):

        out = {}
        out['is_det'] = False # indicating this is from detecton data format in TSV

        image_id = self.image_ids[index]
        out['data_id'] = image_id
        import pdb; pdb.set_trace()
        # Image 
        file_name = self.image_id_to_filename[image_id]
        image = self.fetch_image(file_name)
        out["image"] = image
        out["file_name"] = file_name
    
        # No caption, you need to create one using categories name on fly in TSV 
        
        # For COCO
        # import pdb; pdb.set_trace()
        annIds = self.coco_caps.getAnnIds(imgIds=image_id)
        captions = self.coco_caps.loadAnns(annIds) #self.data[data_id]["caption"]
        captions_text = [cap['caption'] for cap in captions]
        out["caption"] = captions_text
        
        annos = deepcopy(self.image_id_to_objects[image_id])
        for anno in annos:
            anno['category_name'] = self.object_idx_to_name[ anno['category_id'] ]
            
            anno['text_embedding_before'] =  torch.load( os.path.join(self.annotation_embedding_path,"text_features_before",str(anno["id"])), map_location='cpu'  )
            
            anno['image_embedding_before'] = torch.load( os.path.join(self.annotation_embedding_path,"image_features_before",str(anno["id"])), map_location='cpu'  )

            anno['text_embedding_after'] =  torch.load( os.path.join(self.annotation_embedding_path,"text_features_after",str(anno["id"])), map_location='cpu'  )

            anno['image_embedding_after'] = torch.load( os.path.join(self.annotation_embedding_path,"image_features_after",str(anno["id"])), map_location='cpu'  )
            
            # add ref mask:
            # import pdb; pdb.set_trace()
            anno['ref_mask'] = torch.load( os.path.join(self.annotation_embedding_path,"ref_mask",str(anno["id"])), map_location='cpu'  )
            anno['ref_img'] = torch.load( os.path.join(self.annotation_embedding_path,"ref_img",str(anno["id"])), map_location='cpu'  )
            anno['ref_box'] = torch.load( os.path.join(self.annotation_embedding_path,"ref_box",str(anno["id"])), map_location='cpu'  )
            # print(anno['ref_mask'].shape)
        out['annos'] = annos

        return out 


    def __len__(self):
        return len(self.image_ids)
	



class CDDataset(Base):
    "This only supports instance_json, thus only for O365"
    def __init__(self, image_root, instances_json_path, annotation_embedding_path):
        super().__init__(image_root)

        self.image_root = image_root
        self.instances_json_path = instances_json_path
        self.annotation_embedding_path = annotation_embedding_path
        

        # Load all jsons 
        with open(instances_json_path, 'r') as f:
            instances_data = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
        clean_annotations(instances_data["annotations"])
        self.instances_data = instances_data

        # Misc  
        self.image_ids = [] # main list for selecting images
        self.image_id_to_filename = {} # file names used to read image
        for image_data in self.instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename

        
        # All category names (including things and stuff)
        self.object_idx_to_name = {} 
        for category_data in self.instances_data['categories']:
            self.object_idx_to_name[category_data['id']] = category_data['name']


        # Add object data from instances and stuff 
        self.image_id_to_objects = defaultdict(list)
        for object_anno in self.instances_data['annotations']:
            image_id = object_anno['data_id']
            self.image_id_to_objects[image_id].append(object_anno)
        

    def __getitem__(self, index):

        out = {}
        out['is_det'] = True # indicating this is from detecton data format in TSV

        image_id = self.image_ids[index]
        out['data_id'] = image_id
        
        # Image 
        file_name = self.image_id_to_filename[image_id]
        image = self.fetch_image(file_name)
        out["image"] = image
        out["file_name"] = file_name
    
        # No caption, you need to create one using categories name on fly in TSV 
        
        # For COCO
        # out["caption"] = self.data[data_id]["caption"]
        # import pdb; pdb.set_trace()
        annos = deepcopy(self.image_id_to_objects[image_id])
        for anno in annos:
            anno['category_name'] = self.object_idx_to_name[ anno['category_id'] ]

            anno['text_embedding_before'] =  torch.load( os.path.join(self.annotation_embedding_path,"text_features_before",str(anno["id"])), map_location='cpu'  )

            anno['image_embedding_before'] = torch.load( os.path.join(self.annotation_embedding_path,"image_features_before",str(anno["id"])), map_location='cpu'  )

            anno['text_embedding_after'] =  torch.load( os.path.join(self.annotation_embedding_path,"text_features_after",str(anno["id"])), map_location='cpu'  )

            anno['image_embedding_after'] = torch.load( os.path.join(self.annotation_embedding_path,"image_features_after",str(anno["id"])), map_location='cpu'  )
            
            # add ref mask:
            # anno['ref_mask'] = torch.load( os.path.join(self.annotation_embedding_path,"ref_mask",str(anno["id"])), map_location='cpu'  )
            anno['ref_mask'] = torch.load( os.path.join(self.annotation_embedding_path,"ref_mask",str(anno["id"])), map_location='cpu'  )
            anno['ref_img'] = torch.load( os.path.join(self.annotation_embedding_path,"ref_img",str(anno["id"])), map_location='cpu'  )
            anno['ref_box'] = torch.load( os.path.join(self.annotation_embedding_path,"ref_box",str(anno["id"])), map_location='cpu'  )
        out['annos'] = annos

        return out 


    def __len__(self):
        return len(self.image_ids)
	


def split_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



if __name__ == "__main__":
    import argparse
    import math
    parser = argparse.ArgumentParser()
    parser.add_argument("--which_dataset", type=str, default="grounding", help="grounding is for GoldG, CC3M, SBU etc, detection is for O365")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--total_chunk", type=int, default=1)
    parser.add_argument("--image_root", type=str)
    parser.add_argument("--json_path", type=str)
    parser.add_argument("--annotation_embedding_path", type=str, help='offline processed feature embedding from process_grounding.py script')
    parser.add_argument("--tsv_path", type=str)
    args = parser.parse_args()
    assert args.which_dataset in ["grounding", "detection","coco"]

    image_root = args.image_root
    json_path = args.json_path
    annotation_embedding_path = args.annotation_embedding_path
    tsv_path = args.tsv_path

    # image_root = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/SBU/images/"  # path to image zip file or a image folder 
    # json_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/SBU/anno.json"  # json annotation used by my normal dataset
    # annotation_embedding_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/SBU/embedding_clip/" # it must contain 'image_features' and 'text_features'
    # tsv_path = f"/nobackup3/yuheng-data/diffusion_few_shot/DATA/GROUNDING/SBU/tsv/train-{args.chunk_idx:02d}.tsv"

    # image_root = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/CC3M/images/"  # path to image zip file or a image folder 
    # json_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/CC3M/anno.json"  # json annotation used by my normal dataset
    # annotation_embedding_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/CC3M/embedding_clip/" # it must contain 'image_features' and 'text_features'
    # tsv_path = f"/nobackup3/yuheng-data/diffusion_few_shot/DATA/GROUNDING/CC3M/tsv/train-{args.chunk_idx:02d}.tsv"

    # image_root = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/flickr30k/flickr30k_images/"  # path to image zip file or a image folder 
    # json_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/flickr30k/final_flickr_separateGT_train.json"  # json annotation used by my normal dataset
    # annotation_embedding_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/flickr30k/embedding_clip/" # it must contain 'image_features' and 'text_features'
    # tsv_path = f"/nobackup3/yuheng-data/diffusion_few_shot/DATA/GROUNDING/flickr30k/tsv/train-{args.chunk_idx:02d}.tsv"

    # image_root = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/gqa/images/"  # path to image zip file or a image folder 
    # json_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/gqa/final_mixed_train_vg.json"  # json annotation used by my normal dataset
    # annotation_embedding_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/gqa/embedding_clip/" # it must contain 'image_features' and 'text_features'
    # tsv_path = f"/nobackup3/yuheng-data/diffusion_few_shot/DATA/GROUNDING/gqa/tsv/train-{args.chunk_idx:02d}.tsv"


    # image_root = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/OBJECTS365/images.zip"  # path to image zip file or a image folder 
    # json_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/OBJECTS365/instances_train.json"  # json annotation used by my normal dataset
    # annotation_embedding_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/OBJECTS365/embedding_clip" # it must contain 'image_features' and 'text_features'
    # tsv_path = f"/nobackup3/yuheng-data/diffusion_few_shot/DATA/OBJECTS365/tsv/train-{args.chunk_idx:02d}.tsv"

    if args.which_dataset == "coco":
        dataset = COCODataset(image_root,json_path, annotation_embedding_path)
    elif args.which_dataset == "grounding":
        dataset = GroundingDataset(image_root,json_path, annotation_embedding_path)
    else:
        dataset = CDDataset(image_root,json_path, annotation_embedding_path)

    # dataset.getitem(1)
    N = len(dataset)
    print(f'{N} items in total')

    chunk_size = math.ceil(N / args.total_chunk)
    indices = list(split_chunks(list(range(N)), chunk_size))[args.chunk_idx]

    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
    writer = TSVWriter(tsv_path)

    for i in tqdm(indices):
        # import pdb; pdb.set_trace()
        item = dataset[i]
        # import pdb; pdb.set_trace()
        item = item_to_encodable(item)
        row = [item['data_id'], json.dumps(item)]
        writer.write(row)

    writer.close()
