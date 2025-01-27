import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq


# wav2vec-aasistのパラメーターを固定して、全結合層を1つはさむ
class CombinedClassifier_1layer_add(nn.Module):
    def __init__(self, ast_model_vector160, wav2vec_model):
        super(CombinedClassifier_1layer_add, self).__init__()
        self.wav2vec = wav2vec_model  # wav2vec_modelインスタンス
        self.ast = ast_model_vector160  # astモデルのインスタンス化
        #self.fc = torch.nn.Linear(928, 2)  # 結合後の全結合層
        
        combined_dim = self.wav2vec.output_dim + self.ast.output_dim
        self.fc = nn.Linear(combined_dim, 250) # 全結合層を動的に初期化
        self.fc2 = nn.Linear(250, 2)

        # 1つ目のステップで保存したWav2VecClassifierから、wav2vec部分だけをロード
        pretrained_dict = torch.load('/home/haruto/SSL_Anti-spoofing/LA_model.pth')
        # Model_wav2vec に必要なパラメータだけをフィルタリング
        model_dict = self.wav2vec.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'out_layer' not in k}

        # フィルタリングされたパラメータをモデルにロード
        model_dict.update(filtered_dict)
        self.wav2vec.load_state_dict(model_dict)

        # wav2vec_modelのパラメータを固定
        for param in self.wav2vec.parameters():
            param.requires_grad = False

    def forward(self, wav2vec_input, ast_input):
        #print(self.ast.output_dim)
        #print(self.wav2vec.output_dim)
        x1 = self.wav2vec(wav2vec_input)  # wav2vecで特徴量を抽出
        x2 = self.ast(ast_input)  # astモデルで特徴量を抽出
        x = torch.cat((x1, x2), dim=1)  # 特徴量を結合
        x = self.fc(x)  # 全結合層で分類
        x = self.fc2(x)
        return x


# wav2vec-aasistのパラメーターを固定して、全結合層を2つはさむ
class CombinedClassifier_2layer_add(nn.Module):
    def __init__(self, ast_model_vector160, wav2vec_model):
        super(CombinedClassifier_2layer_add, self).__init__()
        self.wav2vec = wav2vec_model  # wav2vec_modelインスタンス
        self.ast = ast_model_vector160  # astモデルのインスタンス化
        #self.fc = torch.nn.Linear(928, 2)  # 結合後の全結合層
        
        combined_dim = self.wav2vec.output_dim + self.ast.output_dim
        self.fc = nn.Linear(combined_dim, 250) # 全結合層を動的に初期化
        self.fc1 = nn.Linear(250, 50) # 全結合層を動的に初期化
        self.fc2 = nn.Linear(50, 2)

        # 1つ目のステップで保存したWav2VecClassifierから、wav2vec部分だけをロード
        pretrained_dict = torch.load('/home/haruto/SSL_Anti-spoofing/LA_model.pth')
        # Model_wav2vec に必要なパラメータだけをフィルタリング
        model_dict = self.wav2vec.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'out_layer' not in k}

        # フィルタリングされたパラメータをモデルにロード
        model_dict.update(filtered_dict)
        self.wav2vec.load_state_dict(model_dict)

        # wav2vec_modelのパラメータを固定
        for param in self.wav2vec.parameters():
            param.requires_grad = False

    def forward(self, wav2vec_input, ast_input):
        x1 = self.wav2vec(wav2vec_input)  # wav2vecで特徴量を抽出
        x2 = self.ast(ast_input)  # astモデルで特徴量を抽出
        x = torch.cat((x1, x2), dim=1)  # 特徴量を結合
        x = self.fc(x)  # 全結合層で分類
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class CombinedClassifier(nn.Module):
    def __init__(self, ast_model_vector160, wav2vec_model):
        super(CombinedClassifier, self).__init__()
        self.wav2vec = wav2vec_model  # wav2vec_modelインスタンス
        self.ast = ast_model_vector160  # astモデルのインスタンス化
        #self.fc = torch.nn.Linear(928, 2)  # 結合後の全結合層
        
        combined_dim = self.wav2vec.output_dim + self.ast.output_dim
        self.fc = nn.Linear(combined_dim, 2) # 全結合層を動的に初期化

        # 1つ目のステップで保存したWav2VecClassifierから、wav2vec部分だけをロード
        pretrained_dict = torch.load('/home/haruto/SSL_Anti-spoofing/LA_model.pth')
        # Model_wav2vec に必要なパラメータだけをフィルタリング
        model_dict = self.wav2vec.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'out_layer' not in k}

        # フィルタリングされたパラメータをモデルにロード
        model_dict.update(filtered_dict)
        self.wav2vec.load_state_dict(model_dict)

        # wav2vec_modelのパラメータを固定
        for param in self.wav2vec.parameters():
            param.requires_grad = False

    def forward(self, wav2vec_input, ast_input):
        x1 = self.wav2vec(wav2vec_input)  # wav2vecで特徴量を抽出
        x2 = self.ast(ast_input)  # astモデルで特徴量を抽出
        x = torch.cat((x1, x2), dim=1)  # 特徴量を結合
        x = self.fc(x)  # 全結合層で分類
        return x


# ASTとwav2vecの出力を連結して予測するモデル
class MixModel(nn.Module):
    def __init__(self, ast_model_vector160, wav2vec_model, embed_size, num_classes):
        super(MixModel, self).__init__()
        self.ast_model_vector160 = ast_model_vector160  # AST_Vectorのインスタンス
        self.wav2vec_model = wav2vec_model  # Model_wav2vecのインスタンス
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, wav2vec_input, ast_input):
        # AST_VectorモデルとModel_wav2vecモデルのそれぞれの出力（長さ160のベクトル）を取得
        ast_output = self.ast_model_vector160(ast_input)  # (batch_size, 160)
        wav2vec_output = self.wav2vec_model(wav2vec_input)  # (batch_size, 160)

        # 2つのベクトルを連結して、(batch_size, 320) のベクトルを作成
        combined_output = torch.cat((ast_output, wav2vec_output), dim=1)
        output = self.fc(combined_output)
        return output

# ASTとwav2vecの出力を連結して予測するモデル、全結合層を1つはさむ
class MixModel_1layer_add(nn.Module):
    def __init__(self, ast_model_vector160, wav2vec_model, embed_size, num_classes):
        super().__init__()
        self.ast_model_vector160 = ast_model_vector160  # AST_Vectorのインスタンス
        self.wav2vec_model = wav2vec_model  # Model_wav2vecのインスタンス
        self.fc1 = nn.Linear(embed_size, 200)
        self.fc2 = nn.Linear(200, num_classes)

    def forward(self, wav2vec_input, ast_input):
        # AST_VectorモデルとModel_wav2vecモデルのそれぞれの出力（長さ160のベクトル）を取得
        ast_output = self.ast_model_vector160(ast_input)  # (batch_size, 160)
        wav2vec_output = self.wav2vec_model(wav2vec_input)  # (batch_size, 160)

        # 2つのベクトルを連結して、(batch_size, 320) のベクトルを作成
        combined_output = torch.cat((ast_output, wav2vec_output), dim=1)
        output1 = self.fc1(combined_output)
        output2 = self.fc2(output1)
        return output2

# AST
class AST_Vector(nn.Module):
    def __init__(self, ast_model, embed_size, vector_long):
       super(AST_Vector, self).__init__()
       self.ast_model = ast_model
       self.fc = nn.Linear(embed_size, vector_long)

    def forward(self, input_values):
       outputs = self.ast_model(input_values)
       outputs = outputs.pooler_output
       output_probs = self.fc(outputs)
       return output_probs


# AST ベクトルの長さ調整なし
class AST_Vector_not(nn.Module):
    def __init__(self, ast_model, embed_size, dropout_rate=0.3):
       super(AST_Vector_not, self).__init__()
       self.ast_model = ast_model
       self.output_dim = 768  # 初期化時には未定義

    def forward(self, input_values):
       outputs = self.ast_model(input_values)
       outputs = outputs.pooler_output
       return outputs
   
# AST ベクトルの長さ調整アリ
class AST_Vector_160_not(nn.Module):
    def __init__(self, ast_model, embed_size, dropout_rate=0.3):
       super(AST_Vector_160_not, self).__init__()
       self.ast_model = ast_model
       self.output_dim = 160  # 初期化時には未定義
       self.fc = nn.Linear(embed_size, 160)

    def forward(self, input_values):
       outputs = self.ast_model(input_values)
       outputs = outputs.pooler_output
       outputs = self.fc(outputs)
       return outputs


# AST ベクトルの長さ調整なし
class AST_Vector_not_dropout(nn.Module):
    def __init__(self, ast_model, embed_size, dropout_rate=0.3):
       super(AST_Vector_not_dropout, self).__init__()
       self.ast_model = ast_model
       self.dropout = nn.Dropout(p=dropout_rate)  # ドロップアウトレイヤー追加
       self.output_dim = 768  # 初期化時には未定義

    def forward(self, input_values):
       outputs = self.ast_model(input_values)
       outputs = outputs.pooler_output
       outputs = self.dropout(outputs)  # ドロップアウトを適用
       
       # output_dim を outputs の形状に基づいて設定
       if self.output_dim is None or self.output_dim != outputs.shape[-1]:
            self.output_dim = outputs.shape[-1]  # バッチ次元を除く特徴量の次元
       return outputs


# FOR fine-tuned SSL MODEL
class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        cp_path = '/home/haruto/xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):      
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()
        
        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb


#---------AASIST back-end------------------------#
''' Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, Hye-jin Shim, Joon Son Chung, Bong-Jin Lee, Ha-Jin Yu and Nicholas Evans. 
    AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks. 
    In Proc. ICASSP 2022, pp: 6367--6371.'''


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        #print('x1',x1.shape)
        #print('x2',x2.shape)
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        #print('num_type1',num_type1)
        #print('num_type2',num_type2)
        x1 = self.proj_type1(x1)
        #print('proj_type1',x1.shape)
        x2 = self.proj_type2(x2)
        #print('proj_type2',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        #print('Concat x1 and x2',x.shape)
        
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
            #print('master',master.shape)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        #print('master',master.shape)
        # directional edge for master node
        master = self._update_master(x, master)
        #print('master',master.shape)
        # projection
        x = self._project(x, att_map)
        #print('proj x',x.shape)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        #print('x1',x1.shape)
        x2 = x.narrow(1, num_type1, num_type2)
        #print('x2',x2.shape)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board


        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h




class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        #print('out',out.shape)
        out = self.conv1(x)

        #print('aft conv1 out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        #print('conv2 out',out.shape)
        
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        #out = self.mp(out)
        return out


class Model_wav2vec(nn.Module):
    def __init__(self, args,device):
        super().__init__()
        self.output_dim = 160
        
        self.device = device
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures =  [2.0, 2.0, 100.0, 100.0]


        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1,1)),
            
        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        
        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer 
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        
        #self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim)
        
        # post-processing on front-end features
        x = x.transpose(1, 2)   #(bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1) # add channel 
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        
        w = self.attention(x)
        
        #------------SA for spectral feature-------------#
        w1 = F.softmax(w,dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S 
        
        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)
        
        #------------SA for temporal feature-------------#
        w2 = F.softmax(w,dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
     
        e_T = m1.transpose(1, 2)
       
        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        
        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        
        last_hidden = self.drop(last_hidden) #多分、長さが160のベクトル
        #output = self.out_layer(last_hidden)
        
        return last_hidden
    


class Model(nn.Module):
    def __init__(self, args,device):
        super().__init__()
        self.device = device
        
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures =  [2.0, 2.0, 100.0, 100.0]


        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1,1)),
            
        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        
        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer 
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim)
        
        # post-processing on front-end features
        x = x.transpose(1, 2)   #(bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1) # add channel 
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        
        w = self.attention(x)
        
        #------------SA for spectral feature-------------#
        w1 = F.softmax(w,dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S 
        
        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)
        
        #------------SA for temporal feature-------------#
        w2 = F.softmax(w,dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
     
        e_T = m1.transpose(1, 2)
       
        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        
        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        
        return output