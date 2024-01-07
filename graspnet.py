import os
import sys
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import pointnet2.pytorch_utils as pt_utils
from models.backbone_resunet14 import MinkUNet14D
from models.modules import *
# ApproachNet, GraspableNet, CloudCrop, SWADNet, SKConv
from loss_utils import GRASP_MAX_WIDTH, NUM_VIEW, NUM_ANGLE, NUM_DEPTH, GRASPNESS_THRESHOLD, M_POINT
from label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from pointnet2.pointnet2_modules import *
import pointnet2_utils


class GraspNet(nn.Module):
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = NUM_DEPTH
        self.num_angle = NUM_ANGLE
        self.M_points = M_POINT
        self.num_view = NUM_VIEW

        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        
        self.rotation = ApproachNet(self.num_view, seed_feature_dim=512, is_training=self.is_training)
        
        
        self.crop1 = CloudCrop(nsample=16, cylinder_radius=cylinder_radius* 0.5, seed_feature_dim=self.seed_feature_dim)
        self.crop2 = CloudCrop(nsample=16, cylinder_radius=cylinder_radius* 0.75, seed_feature_dim=self.seed_feature_dim)
        self.crop3 = CloudCrop(nsample=32, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
        self.crop4 = CloudCrop(nsample=64, cylinder_radius=cylinder_radius* 1.5, seed_feature_dim=self.seed_feature_dim)
        
        
        self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)
        
        self.psa = PSAModule(1024, 256)    
        


    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']  # use all sampled point cloud, B*Ns*3
        # print("xyz", seed_xyz.shape)
        # print(end_points.keys())  
        # dict_keys(['coors', 'feats', 'quantize2original', 'point_clouds', 'graspness_label', 'objectness_label', 
        # 'object_poses_list', 'grasp_points_list', 'grasp_widths_list', 'grasp_scores_list'])
        B, point_num, _ = seed_xyz.shape  # batch _size
        # point-wise features
        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.backbone(mink_input).F
        # print(seed_features.shape)
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)

        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim
        objectness_score = end_points['objectness_score']
        # objectness_score = F.softmax(objectness_score, 1)
        graspness_score = end_points['graspness_score'].squeeze(1)
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        # print("objectness_mask",torch.sum(objectness_mask, -1))  # graspable_mask torch.Size([1, 20000])
        graspness_mask = graspness_score > GRASPNESS_THRESHOLD  # *0.8
        # print("graspness_mask",torch.sum(graspness_mask, -1))
        graspable_mask = objectness_mask & graspness_mask
        # print("graspable_mask",torch.sum(graspable_mask))  # graspable_mask torch.Size([1, 20000])

        seed_features_graspable = []
        seed_xyz_graspable = []
        seed_features_graspable_copy_768 = []

        graspable_num_batch = 0.
        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim
            cur_seed_xyz = seed_xyz[i][cur_mask]  # Ns*3

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1*Ns*3
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
            
            # cur_feat_1 = cur_feat.unsqueeze(0)
            
            # xyz, features_111, fps_idx_1= self.sa_111(cur_seed_xyz, cur_feat_1.transpose(1,2).contiguous())   # 256 channel                        
            # features = pointnet2_utils.gather_operation(cur_feat_1.transpose(1, 2).contiguous(), fps_idx_1)            
            # features_768 = torch.cat([features, features_111], dim=1)  # 把原始特征放前面
            
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # Ns*3
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*Ns
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*Ns

            seed_features_graspable.append(cur_feat.squeeze(0).contiguous())
            seed_xyz_graspable.append(cur_seed_xyz.squeeze(0).contiguous())
            # seed_features_graspable_copy_768.append(features_768.squeeze(0).contiguous())
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*Ns*3
        seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*Ns
        # seed_features_graspable_768 = torch.stack(seed_features_graspable_copy_768)
        
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['fp2_features'] = seed_features_graspable  # 这个有神魔用？？？
        end_points['graspable_count_stage1'] = graspable_num_batch / B


        end_points, res_feat = self.rotation(seed_features_graspable, end_points)   
        # seed_features_graspable = seed_features_graspable + res_feat  # 原始程序里有，这里注释掉了

        if self.is_training:
            end_points = process_grasp_labels(end_points)
            # print(end_points.keys())
            # print("batch_grasp_view_rot", end_points["batch_grasp_view_rot"].shape)  # [4, 1024, 300, 3, 3]
            # print("batch_grasp_view_graspness", end_points['batch_grasp_view_graspness'][0, 0, :])  # [4, 1024, 300]
            # print("batch_grasp_view_graspness", torch.max(end_points['batch_grasp_view_graspness'], -1)[1][0,0])  # [4, 1024]
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)  # 训练时是在所有视角中按照按概率选择，测试时是选择最优的
            
            # top_view_inds_lable = torch.max(end_points['batch_grasp_view_graspness'], -1)[1]
     
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']
            
        end_points['grasp_top_view_rot_pred'] = grasp_top_views_rot
        
        group_features1 = self.crop1(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)  
        group_features2 = self.crop2(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot) 
        group_features3 = self.crop3(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot) 
        group_features4 = self.crop4(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
         
        group_features_concat = torch.cat([group_features1, group_features2, group_features3, group_features4], dim=1)
        # print(group_features_concat.shape)  # torch.Size([4, 1024, 1024])
        group_features = self.psa(group_features_concat)  # torch.Size([4, 256, 1024])
        # print(group_features.shape)
        end_points = self.swad(group_features, end_points)

        return end_points



def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()  # 1024, 12,  4
        grasp_score = end_points['grasp_score_pred'][i].float()  
        # print(grasp_score.shape)  # [1024, 12, 4]
        device = grasp_score.device

        # min_score = torch.min(grasp_score, 1)[0]
        # min_score = torch.min(min_score, 1)[0]
        # print(min_score.shape)
        # min_score = min_score.unsqueeze(-1).unsqueeze(-1).repeat(1, 12, 4).to(device)
        # print(min_score.shape)
        
        zero = torch.zeros(grasp_score.shape).to(device)

        # *******************************************************************************
        grasp_collision_pred = end_points['grasp_collision_pred'][i].float()
        # print(grasp_collision_pred[0, 0, :, :])
        # print(grasp_collision_pred.shape)  # [2, 1024, 12, 4]
        collision_pred = torch.argmax(grasp_collision_pred ,0)
        # print(collision_pred.shape)
        collision_mask = collision_pred == 0
        
        # grasp_score[collision_mask] = min_score[collision_mask]
        grasp_score[collision_mask] = zero[collision_mask]
        # print(collision_pred.shape)
        # one = torch.ones(grasp_score.shape).to(device) 
        #grasp_collision_pred = one - grasp_collision_pred
        #grasp_score = grasp_score * grasp_collision_pred   # 加的东西
        # *******************************************************************************
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)
        grasp_angle = (grasp_score_inds // NUM_DEPTH) * np.pi / 12
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)

        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)

        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M_POINT, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds




"""

def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()
        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)
        grasp_angle = (grasp_score_inds // NUM_DEPTH) * np.pi / 12
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)
        approaching = -end_points['grasp_top_view_xyz'][i].float()
        # print("**************************************")
        # print("approaching", approaching[0])   # [1024, 3]
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        # print("grasp_rot", grasp_rot[0])
        grasp_rot = grasp_rot.view(M_POINT, 9)   # [1024, 3, 3]
        # print("grasp_rot", grasp_rot[0])   # [1024, 9]  旋转矩阵是正交矩阵且各列的模为1

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds
"""




