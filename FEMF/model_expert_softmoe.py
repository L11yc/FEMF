import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from MultiAttn import MultiAttnModel
from modules.Attention_softmoe import *


class DisentangleModule(nn.Module):
    def __init__(self, input_dim, shared_dim_ratio=0.5):
        super(DisentangleModule, self).__init__()
        shared_dim = int(input_dim * shared_dim_ratio)
        private_dim = input_dim - shared_dim

        # 一个统一的共享特征编码器
        self.shared_proj = nn.Linear(input_dim, shared_dim)

        # 各模态独立的私有特征编码器
        self.private_proj_a = nn.Linear(input_dim, private_dim)
        self.private_proj_t = nn.Linear(input_dim, private_dim)
        self.private_proj_v = nn.Linear(input_dim, private_dim)

        # 统一的小Decoder，用于重构
        self.decoder = nn.Sequential(
            nn.Linear(shared_dim + private_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, feat_a, feat_t, feat_v):
        shared_a = self.shared_proj(feat_a)
        shared_t = self.shared_proj(feat_t)
        shared_v = self.shared_proj(feat_v)

        private_a = self.private_proj_a(feat_a)
        private_t = self.private_proj_t(feat_t)
        private_v = self.private_proj_v(feat_v)

        # 拼接后过Decoder
        feat_a_reconstructed = self.decoder(torch.cat([shared_t, private_a], dim=-1))
        feat_t_reconstructed = self.decoder(torch.cat([shared_t, private_t], dim=-1))
        feat_v_reconstructed = self.decoder(torch.cat([shared_t, private_v], dim=-1))

        private_a_new = self.private_proj_a(feat_a_reconstructed)
        # private_t_new = self.private_proj_t(feat_t_reconstructed)
        private_v_new = self.private_proj_v(feat_v_reconstructed)
        

        return shared_a, private_a, shared_t, private_t, shared_v, private_v, feat_a_reconstructed, feat_t_reconstructed, feat_v_reconstructed,private_a_new,private_v_new



class FourierFilterLayer(nn.Module):
    def __init__(self, hidden_dim, ratio=0.5, beta=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.ratio = ratio
        self.LayerNorm = nn.LayerNorm(hidden_dim)
    
    def forward(self, input_tensor):
        """
        input_tensor: [B, T, D]
        返回: filtered output, low_pass, high_pass
        """
        B, T, D = input_tensor.shape
        c = int(T * self.ratio)  # 高频截断点
        x_fft = torch.fft.rfft(input_tensor, dim=1, norm="ortho")  # [B, F, D]

        low_pass = x_fft.clone()
        low_pass[:, c:, :] = 0  # 保留低频
        low_pass = torch.fft.irfft(low_pass, n=T, dim=1, norm="ortho")  # 回到时域 [B, T, D]

        high_pass = input_tensor - low_pass
        sequence_fft = low_pass + (self.beta ** 2) * high_pass

        output = self.LayerNorm(sequence_fft + input_tensor)  # 残差 + 归一化
        return output, low_pass, high_pass



class MoMKE(nn.Module):


    def __init__(self, args, adim, tdim, vdim, D_e, n_classes, depth=4, num_heads=4, mlp_ratio=1, drop_rate=0, attn_drop_rate=0, no_cuda=False):
        super(MoMKE, self).__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        self.num_heads = num_heads
        self.num_layers = 2  #mosei=2,mosi=1
        self.model_dim = 256
        self.hidden = 16
        
        self.embedding_dim = self.model_dim
        D = 3 * D_e

        self.device = args.device
        self.no_cuda = no_cuda
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.out_dropout = args.drop_rate
        self.multiattn = MultiAttnModel(self.num_layers, self.model_dim, num_heads, self.hidden, drop_rate)
        
        # 加入模态共享-私有分离模块
        self.disentangle_module = DisentangleModule(D_e)
        
        # 初始化模块
        self.fourier_filter = FourierFilterLayer(hidden_dim=D_e, ratio=0.9, beta=1.0)


        self.a_in_proj = nn.Sequential(nn.Linear(self.adim, D_e))
        self.t_in_proj = nn.Sequential(nn.Linear(self.tdim, D_e))
        self.v_in_proj = nn.Sequential(nn.Linear(self.vdim, D_e))
        self.dropout_a = nn.Dropout(args.drop_rate)
        self.dropout_t = nn.Dropout(args.drop_rate)
        self.dropout_v = nn.Dropout(args.drop_rate)
        self.proj1_t_high = nn.Linear(D_e, D_e)
        self.proj2_t_high = nn.Linear(D_e, D_e)
            
        self.proj1_v_high = nn.Linear(D_e, D_e)
        self.proj2_v_high = nn.Linear(D_e, D_e)
        
        self.proj1_a_high = nn.Linear(D_e, D_e)
        self.proj2_a_high = nn.Linear(D_e, D_e)

         # 7. project for fusion
        self.projector_t = nn.Linear(D_e, D_e)
        self.projector_v = nn.Linear(D_e, D_e)
        self.projector_a = nn.Linear(D_e, D_e)


        self.projector_c = nn.Linear(D, D)
        
        self.proj_distangle = nn.Linear(128, D_e)
        

        
        self.block_low = Block(
                    dim=D_e,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    depth=depth,
                )

        self.block_high = Block(
                    dim=D_e,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    depth=depth,
                )
        



        self.proj1 = nn.Linear(2 * D, 2 * D)
        # self.proj1 = nn.Linear(D, D)
        self.nlp_head_a = nn.Linear(D_e, n_classes)
        self.nlp_head_t = nn.Linear(D_e, n_classes)
        self.nlp_head_v = nn.Linear(D_e, n_classes)
        self.nlp_head = nn.Linear(2 * D, n_classes)
        # self.nlp_head = nn.Linear(D, n_classes)





       
        # # 原来是 in_features=D_e
        router_input_dim = D_e * 2    # 拼接 audio/text、text/text、visual/text

        self.router_a = Mlp(
            in_features=router_input_dim,
            hidden_features=int(router_input_dim * mlp_ratio),
            out_features=3,
            drop=drop_rate,
        )

        self.router_t = Mlp(
            in_features=router_input_dim,
            hidden_features=int(router_input_dim * mlp_ratio),
            out_features=3,
            drop=drop_rate,
        )

        self.router_v = Mlp(
            in_features=router_input_dim,
            hidden_features=int(router_input_dim * mlp_ratio),
            out_features=3,
            drop=drop_rate,
        )
        
    
    



    def forward(self, inputfeats, input_features_mask=None, umask=None, first_stage=False):
        """
        inputfeats -> ?*[seqlen, batch, dim]
        qmask -> [batch, seqlen]cc
        umask -> [batch, seqlen]
        seq_lengths -> each conversation lens
        input_features_mask -> ?*[seqlen, batch, 3]
        """
        # print(inputfeats[:,:,:])
        # print(input_features_mask[:,:,1])

        # weight_save = []
        alpha = 0.7   #mosi=0.3,mosei=0.7
        weight_low_save = []
        weight_high_save = []
        # sequence modeling
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], \
        inputfeats[:, :, self.adim + self.tdim:]
        seq_len, B, C = audio.shape

        # --> [batch, seqlen, dim]
        audio, text, video = audio.permute(1, 0, 2), text.permute(1, 0, 2), video.permute(1, 0, 2)
        proj_a = self.dropout_a(self.a_in_proj(audio))
        proj_t = self.dropout_t(self.t_in_proj(text))
        proj_v = self.dropout_v(self.v_in_proj(video))
        # proj_v = torch.zeros_like(proj_v)
        # proj_t = torch.zeros_like(proj_t)

        # 共享-私有分离
        shared_a, private_a, shared_t, private_t, shared_v, private_v, feat_a_rec, feat_t_rec, feat_v_rec,private_a_new,private_v_new = self.disentangle_module(proj_a, proj_t, proj_v)

        proj_a_new = torch.cat([shared_a, private_a], dim=-1)
        proj_t_new = torch.cat([shared_t, private_t], dim=-1)
        proj_v_new = torch.cat([shared_v, private_v], dim=-1)



        
        # --> [batch, seqlen, 3]
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        # --> [batch, 3, seqlen] -> [batch, 3*seqlen]
        attn_mask = input_mask.transpose(1, 2).reshape(B, -1)

        
        # 拼接模态特征 + 文本特征
        # router_input_a = torch.cat([proj_a, proj_t], dim=-1)  # [B, T, 2*D_e]
        # router_input_t = torch.cat([proj_t, proj_t], dim=-1)
        # router_input_v = torch.cat([proj_v, proj_t], dim=-1)
        x_private_a =  self.proj_distangle(private_a)
        x_private_t =  self.proj_distangle(private_t)
        x_private_v =  self.proj_distangle(private_v)



        

        x_a_filter, x_a_low, x_a_high = self.fourier_filter(x_private_a)
        x_t_filter, x_t_low, x_t_high = self.fourier_filter(x_private_t)
        x_v_filter, x_v_low, x_v_high = self.fourier_filter(x_private_v)

        # x_a_filter, x_a_low, x_a_high = self.fourier_filter(proj_a)
        # x_t_filter, x_t_low, x_t_high = self.fourier_filter(proj_t)
        # x_v_filter, x_v_low, x_v_high = self.fourier_filter(proj_v)

        # print('x_a_low.shape:', x_a_low.shape)
        # print('x_a_high.shape:', x_a_high.shape)

        router_input_a_low = torch.cat([x_a_low, x_t_low], dim=-1)  # [B, T, 2*D_e]
        router_input_t_low = torch.cat([x_t_low, x_t_low], dim=-1)
        router_input_v_low = torch.cat([x_v_low, x_t_low], dim=-1)

        router_input_a_high = torch.cat([x_a_high, x_t_high], dim=-1)  # [B, T, 2*D_e]
        router_input_t_high = torch.cat([x_t_high, x_t_high], dim=-1)
        router_input_v_high = torch.cat([x_v_high, x_t_high], dim=-1)
       
        # weight_a = torch.softmax(self.router_a(router_input_a), dim=-1)
        # weight_t = torch.softmax(self.router_t(router_input_t), dim=-1)
        # weight_v = torch.softmax(self.router_v(router_input_v), dim=-1)

        weight_a_low = torch.softmax(self.router_a(router_input_a_low), dim=-1)
        weight_t_low = torch.softmax(self.router_t(router_input_t_low), dim=-1)
        weight_v_low = torch.softmax(self.router_v(router_input_v_low), dim=-1)

        weight_a_high = torch.softmax(self.router_a(router_input_a_high), dim=-1)
        weight_t_high = torch.softmax(self.router_t(router_input_t_high), dim=-1)
        weight_v_high = torch.softmax(self.router_v(router_input_v_high), dim=-1)

        # weight_save.append(np.array([weight_a.cpu().detach().numpy(), weight_t.cpu().detach().numpy(), weight_v.cpu().detach().numpy()]))
        # weight_a = weight_a.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
        # weight_t = weight_t.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
        # weight_v = weight_v.unsqueeze(-1).repeat(1, 1, 1, self.D_e)

        weight_low_save.append(np.array([weight_a_low.cpu().detach().numpy(), weight_t_low.cpu().detach().numpy(), weight_v_low.cpu().detach().numpy()]))
        weight_a_low = weight_a_low.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
        weight_t_low = weight_t_low.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
        weight_v_low = weight_v_low.unsqueeze(-1).repeat(1, 1, 1, self.D_e)

        weight_high_save.append(np.array([weight_a_high.cpu().detach().numpy(), weight_t_high.cpu().detach().numpy(), weight_v_high.cpu().detach().numpy()]))
        weight_a_high = weight_a_high.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
        weight_t_high = weight_t_high.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
        weight_v_high = weight_v_high.unsqueeze(-1).repeat(1, 1, 1, self.D_e)

        


        # --> [batch, 3*seqlen, dim]
        x_a_low = self.block_low(x_a_low, first_stage, attn_mask, 'a')
        x_t_low = self.block_low(x_t_low, first_stage, attn_mask, 't')
        x_v_low = self.block_low(x_v_low, first_stage, attn_mask, 'v')
        
        x_a_high = self.block_high(x_a_high, first_stage, attn_mask, 'a')
        x_t_high = self.block_high(x_t_high, first_stage, attn_mask, 't')
        x_v_high = self.block_high(x_v_high, first_stage, attn_mask, 'v')

        

        # # --> [batch, 3*seqlen, dim]
        # x_a = self.block(x_private_a, first_stage, attn_mask, 'a')
        # x_t = self.block(x_private_t, first_stage, attn_mask, 't')
        # x_v = self.block(x_private_v, first_stage, attn_mask, 'v')
        

        if first_stage:
            # out_a = self.nlp_head_a(x_a)
            # out_t = self.nlp_head_t(x_t)
            # out_v = self.nlp_head_v(x_v)
            
            # x = torch.cat([x_a, x_t, x_v], dim=1)

            out_a_low = self.nlp_head_a(x_a_low)
            out_t_low = self.nlp_head_t(x_t_low)
            out_v_low = self.nlp_head_v(x_v_low)
            
            x_low = torch.cat([x_a_low, x_t_low, x_v_low], dim=1)

            out_a_high = self.nlp_head_a(x_a_high)
            out_t_high = self.nlp_head_t(x_t_high)
            out_v_high = self.nlp_head_v(x_v_high)
            
            x_high = torch.cat([x_a_high, x_t_high, x_v_high], dim=1)

            x = alpha * x_low + (1-alpha) * x_high
        else:
            # meaningless
            # out_a, out_t, out_v = torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes))

            out_a_low, out_t_low, out_v_low = torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes))
            out_a_high, out_t_high, out_v_high = torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes)), torch.rand((B, seq_len, self.n_classes))

            # x_unweighted_a = x_a.reshape(B, seq_len, 3, self.D_e)
            # x_unweighted_t = x_t.reshape(B, seq_len, 3, self.D_e)
            # x_unweighted_v = x_v.reshape(B, seq_len, 3, self.D_e)
            # x_out_a = torch.sum(weight_a * x_unweighted_a, dim=2)
            # x_out_t = torch.sum(weight_t * x_unweighted_t, dim=2)
            # x_out_v = torch.sum(weight_v * x_unweighted_v, dim=2)

            x_unweighted_a_low = x_a_low.reshape(B, seq_len, 3, self.D_e)
            x_unweighted_t_low = x_t_low.reshape(B, seq_len, 3, self.D_e)
            x_unweighted_v_low = x_v_low.reshape(B, seq_len, 3, self.D_e)

            x_unweighted_a_high = x_a_high.reshape(B, seq_len, 3, self.D_e)
            x_unweighted_t_high = x_t_high.reshape(B, seq_len, 3, self.D_e)
            x_unweighted_v_high = x_v_high.reshape(B, seq_len, 3, self.D_e)

            x_out_a_low = torch.sum(weight_a_low * x_unweighted_a_low, dim=2)
            x_out_t_low = torch.sum(weight_t_low * x_unweighted_t_low, dim=2)
            x_out_v_low = torch.sum(weight_v_low * x_unweighted_v_low, dim=2)
            
            # print('x_out_a_low.shape:', x_out_a_low.shape)

            x_out_a_high = torch.sum(weight_a_high * x_unweighted_a_high, dim=2)
            x_out_t_high = torch.sum(weight_t_high * x_unweighted_t_high, dim=2)
            x_out_v_high = torch.sum(weight_v_high * x_unweighted_v_high, dim=2)
            
            # print('x_out_a_high.shape:', x_out_a_high.shape)
            
            x_low = torch.cat([x_out_a_low, x_out_t_low, x_out_v_low], dim=1)
            x_high = torch.cat([x_out_a_high, x_out_t_high, x_out_v_high], dim=1)
            
            x = alpha * x_low + (1-alpha) * x_high

            

            # x = torch.cat([x_out_a, x_out_t, x_out_v], dim=1)
        
        
        x[attn_mask == 0] = 0
        # print('x.shape:', x.shape)
        x_a, x_t, x_v = x[:, :seq_len, :], x[:, seq_len:2*seq_len, :], x[:, 2*seq_len:, :]

        # print('x_a.shape:', x_a.shape)
        # print('x_t.shape:', x_t.shape)
        # print('x_v.shape:', x_v.shape)

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x_fina_a, x_fina_t, x_fina_v = self.multiattn(x_a, x_t, x_v)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # x_fina_a, x_fina_t, x_fina_v = self.multiattn(x_private_a, x_private_t, x_private_v)

        hs_proj_a_high = self.proj2_a_high(F.dropout(F.relu(self.proj1_a_high(x_fina_a), inplace=True), p=self.out_dropout, training=self.training))
        hs_proj_a_high += x_fina_a

        hs_proj_v_high = self.proj2_v_high(F.dropout(F.relu(self.proj1_v_high(x_fina_v), inplace=True), p=self.out_dropout, training=self.training))
        hs_proj_v_high += x_fina_v
        
        hs_proj_t_high = self.proj2_t_high(F.dropout(F.relu(self.proj1_t_high(x_fina_t), inplace=True), p=self.out_dropout, training=self.training))
        hs_proj_t_high += x_fina_t

        # hs_proj_a_high = self.proj2_a_high(F.dropout(F.relu(self.proj1_a_high(x_a), inplace=True), p=self.out_dropout, training=self.training))
        # hs_proj_a_high += x_a

        # hs_proj_v_high = self.proj2_v_high(F.dropout(F.relu(self.proj1_v_high(x_v), inplace=True), p=self.out_dropout, training=self.training))
        # hs_proj_v_high += x_v
        
        # hs_proj_t_high = self.proj2_t_high(F.dropout(F.relu(self.proj1_t_high(x_t), inplace=True), p=self.out_dropout, training=self.training))
        # hs_proj_t_high += x_t

        predict_t = torch.sigmoid(self.projector_t(hs_proj_t_high))   
        predict_v = torch.sigmoid(self.projector_v(hs_proj_v_high))
        predict_a = torch.sigmoid(self.projector_a(hs_proj_a_high))

        shared_atv = shared_a + shared_v + shared_t
        shared_atv_proj = self.proj_distangle(shared_atv)
        shared_predict = self.nlp_head_t(shared_atv_proj)
        

        
        x_joint = torch.cat([x_fina_a, x_fina_t, x_fina_v], dim=-1)
        # x_joint = torch.cat([x_a, x_t, x_v], dim=-1)

        x_fusion = torch.sigmoid(self.projector_c(x_joint))
        last_hs = torch.cat([predict_t, predict_v, predict_a, x_fusion], dim=-1)
    
        out_final_a = self.nlp_head_a(x_fina_a)
        out_final_t = self.nlp_head_t(x_fina_t)
        out_final_v = self.nlp_head_v(x_fina_v)
        # out_final_a = self.nlp_head_a(x_a)
        # out_final_t = self.nlp_head_t(x_t)
        # out_final_v = self.nlp_head_v(x_v)

        # res = x_fusion
        # u = F.relu(self.proj1(x_fusion))
        # u = F.dropout(u, p=self.out_dropout, training=self.training)
        # hidden = u + res
        # out = self.nlp_head(hidden)

        # gmma = torch.softmax(torch.mlp(last_hs))

        res = last_hs
        u = F.relu(self.proj1(last_hs))
        u = F.dropout(u, p=self.out_dropout, training=self.training)
        hidden = u + res
        out = self.nlp_head(hidden)
        # print('out.shape:', out.shape)

        return hidden, out, out_a_low,out_a_high, out_t_low,out_t_high, out_v_low,out_v_high,out_final_a,out_final_t,out_final_v,shared_predict, np.array(weight_low_save),np.array(weight_high_save), proj_a, proj_t, proj_v, shared_a, private_a, shared_t, private_t, shared_v, private_v, feat_a_rec, feat_t_rec, feat_v_rec,private_a_new,private_v_new



if __name__ == '__main__':
    input = [torch.randn(61, 32, 300)]
    model = MoMKE(100, 100, 100, 128, 1)
    anchor = torch.randn(32, 61, 128)
    hidden, out, _ = model(input)
