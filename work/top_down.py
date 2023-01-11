# Check Pytorch installation
import torch, torchvision

print('torch version:', torch.__version__, torch.cuda.is_available())
print('torchvision version:', torchvision.__version__)

# Check MMPose installation
import mmpose

print('mmpose version:', mmpose.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

print('cuda version:', get_compiling_cuda_version())
print('compiler information:', get_compiler_version())

import cv2
import numpy as np
import json
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

def export_json(dst: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(dst, f, indent=2)


class Pedestrian:
    def __init__(self, token: str, bbox: list) -> None:
        self.token = token
        self.bbox = bbox
    
    def get_bbox_array(self, dummy_acc: float = 1.0) -> np.ndarray:
        include_dummy_acc = self.bbox.copy()
        include_dummy_acc.append(dummy_acc)
        return np.array(include_dummy_acc, dtype=np.float32)

class Pose:
    def __init__(self, pose_config, pose_checkpoint) -> None:
        # initialize pose model
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint
        self.pose_model = init_pose_model(self.pose_config, self.pose_checkpoint)

    def get(self, img_path: str, peds: list) -> list:
        person_results = []
        ped: Pedestrian
        for ped in peds:
            person = {}
            person['token'] = ped.token
            person['bbox'] = ped.get_bbox_array()
            person_results.append(person)

        # inference pose
        pose_results, _ = inference_top_down_pose_model(
            self.pose_model,
            img_path,
            person_results,
            bbox_thr=0.3,
            format='xyxy',
            dataset=self.pose_model.cfg.data.test.type)
        
        return pose_results

    def export(self, pose_results: list, output_path: str) -> None:
        dst = {}
        pose_result: dict
        for pose_result in pose_results:
            tmp = {}
            tmp['bbox'] = pose_result['bbox'].tolist()[:-1]
            tmp['pose'] = pose_result['keypoints'].tolist()
            dst[pose_result['token']] = tmp
        export_json(dst, output_path)

    def render(self, img_path: str, pose_results: list, save_path: str) -> None:
        # show pose estimation results
        vis_result = vis_pose_result(
            self.pose_model,
            img_path,
            pose_results,
            dataset=self.pose_model.cfg.data.test.type,
            show=False)
        # reduce image size
        vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)
        cv2.imwrite(save_path, vis_result)


def main():
    pose_config = '../mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_l_p4_w7_coco_384x288.py'
    pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_384x288-c36b7845_20220705.pth'
    img_path = '../pedestrian/nuimages_ped/v1.0-train/img/00a14431d93f4f32adf683e139dfdf94.jpg'
    pose = Pose(pose_config, pose_checkpoint)

    token = ["ped1", "ped2", "ped3"]
    bboxes = [
        [1318, 447, 1396, 672],
        [1416, 440, 1503, 703],
        [1399, 453, 1520, 684]
    ]
    peds = []
    for i in range(len(token)):
        peds.append(Pedestrian(token[i], bboxes[i]))

    pose_results = pose.get(img_path, peds)
    pose.render(img_path, pose_results)
    pose.export(pose_results, 'output.json')

if __name__ == "__main__":
    main()