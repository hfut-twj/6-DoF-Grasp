import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import pointnet2.pytorch_utils as pt_utils
from pointnet2.pointnet2_utils import CylinderQueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix



class GraspableNet(nn.Module):  # end_points = self.graspable(seed_features, end_points)
    def __init__(self, seed_feature_dim):   # 从512维直接降到3维？？？
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)  # (B, 3, num_seed)
        end_points['objectness_score'] = graspable_score[:, :2]  # 这个的损失有点大
        end_points['graspness_score'] = graspable_score[:, 2]
        return end_points


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim, is_training=True):
    # self.rotation = ApproachNet(self.num_view=300, seed_feature_dim=self.seed_feature_dim=512, is_training=self.is_training)
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()   # num_seed=1024
         
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        
        features = self.conv2(res_features)
        view_score = features.transpose(1, 2).contiguous() # (B, num_seed, num_view)
        end_points['view_score'] = view_score  # 每一个视角的分数，这里不加一个softmax？？？

        if self.is_training:
            # normalize view graspness score to 0~1
            view_score_ = view_score.clone().detach()
            view_score_max, _ = torch.max(view_score_, dim=2)
            view_score_min, _ = torch.min(view_score_, dim=2)
            view_score_max = view_score_max.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_min = view_score_min.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)

            top_view_inds = []
            for i in range(B):
                top_view_inds_batch = torch.multinomial(view_score_[i], 1, replacement=False)
                
                # 作用是对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标。
                # input张量可以看成一个权重张量，每一个元素代表其在该行中的权重。如果有元素为0，那么在其他不为0的元素被取干净之前，这个元素是不会被取到的。
                # 但非零权重都有可能被取到，权重大的概率大
                top_view_inds.append(top_view_inds_batch)
            top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(-1)  # B, num_seed
            # print("top_view_inds", top_view_inds.shape)  # [B 1024]
            # print(top_view_inds[:, 0:100])
            
        else:
            _, top_view_inds = torch.max(view_score, dim=2)  # (B, num_seed)
            # print("**************************************")
            # print("top_view_inds", top_view_inds.shape)  # [4, 1024]
            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            # print("top_view_inds_", top_view_inds_.shape)  # [4, 1024, 1, 3]     
            # 生成模板view
            template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
            # print("template_views", template_views.shape)  # [300, 3]
            template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()
            # print("template_views", template_views.shape)  # [4, 1024, 300, 3]
            
            # 确定预测的view在模板view中的分类
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
            # print("vp_xyz", vp_xyz.shape)  # [4, 1024, 3]
            vp_xyz_ = vp_xyz.view(-1, 3)
            # print("vp_xyz_", vp_xyz_.shape)  # [4096, 3]
            
            # 没有旋转
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            # print("batch_angle", batch_angle.shape)  # [4096]
            
            # 将approach转换成3x3的矩阵
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            # print("vp_rot", vp_rot.shape)  #  [4, 1024, 3, 3]
            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        return end_points, res_features


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=1):  # reduction就是为了减少计算量
        super(SEWeightModule, self).__init__()
        # self.conv = nn.Conv1d(1024, 256, 1)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)  # 这里加上会有效果吗？？？
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # torch.Size([1, 64, 1024, 4])
        # print("x", x.shape)
        # out = self.fc1(out)
        # out = self.relu(out)
        # out = self.fc2(out)
        weight = self.sigmoid(x)
        # print("weight", weight.shape)

        return weight


class PSAModule(nn.Module):
    # inplans是输入的特征维度，planes数想要输出的特征维度
    def __init__(self, inplans, planes):
        super(PSAModule, self).__init__()
        self.se = SEWeightModule(planes // 4)   # planes // 4的值是64
        self.conv = nn.Conv1d(1024, 256, 1)
        self.split_channel = planes // 4  # 这就是要输出的特征的维度
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # x的shape是(1, 1024, 1024)
        batch_size = x.shape[0]
        # print("原始", x.shape)
        x = x.view(batch_size, 1024,  -1)
        # print("变换前", x.shape)
        x = self.conv(x)
        # print("变换后", x.shape)
        x = x.view(batch_size, 4, -1, 1024)
        # print("变换维度", x.shape)

        x1 = x[:, 0]   # 这里本来是1
        # print("x1", x1.shape)  # torch.Size([1, 64, 1024])
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # feats就是卷积之后的特征
        feats = torch.cat((x1, x2, x3, x4), dim=1)  # 原始特征分割以及聚合完成
        # print("feats", feats.shape)
        feats = feats.view(batch_size, 4, self.split_channel, 1024)
        # print("feats", feats.shape)

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)
        # print("x4_se", x4_se.shape)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        # print("x_se", x_se.shape)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1024)
        # print("attention_vectors", attention_vectors.shape)
        attention_vectors = self.softmax(attention_vectors)
        # print("attention_vectors", attention_vectors.shape)

        # print(feats[0,:,0,0])
        # print(attention_vectors[0, :, 0, 0])
        feats_weight = feats * attention_vectors
        # ("feats_weight", feats_weight.shape)

        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)
        # print("out", out.shape)
        return out




class CloudCrop(nn.Module):
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3
        self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                             use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot,
                                       seed_features_graspable)  # B*3 + feat_dim*M*K
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features




class SWADNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle  # 12
        self.num_depth = num_depth  # 4

        self.conv1 = nn.Conv1d(256, 256, 1)  # input feat dim need to be consistent with CloudCrop module
        self.conv_swad = nn.Conv1d(256, 4*num_angle*num_depth, 1)

    def forward(self, vp_features, end_points):
        B, _, num_seed = vp_features.size()
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        vp_features = self.conv_swad(vp_features)
        vp_features = vp_features.view(B, 4, self.num_angle, self.num_depth, num_seed)
        vp_features = vp_features.permute(0, 1, 4, 2, 3)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0]  # B * num_seed * num angle * num_depth
        end_points['grasp_width_pred'] = vp_features[:, 1]
        end_points['grasp_collision_pred'] = vp_features[:, 2:4]
        
        # end_points['grasp_collision_pred'] = nn.Softmax(1)(end_points['grasp_collision_pred'])
        # end_points['grasp_collision_pred'] = torch.max(end_points['grasp_collision_pred'], 1)[0]
        
        return end_points



        
        print
