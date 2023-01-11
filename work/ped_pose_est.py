import json
from glob import glob
from os import path as osp
import os
from tqdm import tqdm

from top_down import Pose, Pedestrian

def read_json(json_path: str) -> list:
    with open(json_path, 'r') as f:
        _data = json.load(f)
    return _data

def loop_ped(pose: Pose, dataset_name: str, version_name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    not_found_list = []
    for img_path in tqdm(glob('../pedestrian/'+dataset_name+'/'+version_name+'/img/*'), dataset_name+'/'+version_name+' : '):
        _type = img_path[-3:]
        if _type not in ('jpg', 'png'):
            print("image not found")
            continue
        record_token = osp.splitext(osp.basename(img_path))[0]
        json_path = '../annotation/'+dataset_name+'/'+version_name+'/'+record_token+'.json'
        if not osp.isfile(json_path):
            not_found_list.append(json_path)
            continue
        
        peds = read_json(json_path)
        formatted_peds = []
        for token, ann in peds.items():
            formatted_peds.append(Pedestrian(token, ann))
        pose_results = pose.get(img_path, formatted_peds)
        output_path = save_dir+'/'+record_token+'.json'
        pose.export(pose_results, output_path)
        output_path = save_dir+'/'+record_token+'.png'
        pose.render(img_path, pose_results, output_path)
    print(str(len(not_found_list))+"json files not found")
    print(not_found_list)

def main():
    pose_config = '../mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_l_p4_w7_coco_384x288.py'
    pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_384x288-c36b7845_20220705.pth'
    img_path = '../pedestrian/nuimages_ped/v1.0-train/img/00a14431d93f4f32adf683e139dfdf94.jpg'
    pose = Pose(pose_config, pose_checkpoint)

    for dataset in glob('../pedestrian/*'):
        dataset_name = osp.basename(dataset)
        for version in glob(dataset+'/*'):
            version_name = osp.basename(version)
            loop_ped(pose, dataset_name, version_name, '../ped_pose_render/'+dataset_name+'/'+version_name)

if __name__ == "__main__":
    main()