import time
import datetime
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score
import sys
from utils import Logger, get_loaders, build_model, generate_mask, generate_inputs
from loss import MaskedCELoss, MaskedMSELoss
from loss import semantic_agreement_loss_mse_teacher
from loss import orthogonality_loss
import torch.nn.functional as F



import os
import warnings
sys.path.append('./')
warnings.filterwarnings("ignore")
import config
def apply_random_mask(x, mask_ratio=0.2):
    """
    对输入特征 x 添加随机 mask 噪声。
    x: Tensor, 形状 [seq_len, batch_size, dim]
    mask_ratio: 掩蔽的比例，0.2 表示随机 mask 掉 20% 的位置
    """
    # 构造 mask，形状为 [seq_len, batch_size]
    mask = (torch.rand(x.shape[0], x.shape[1], device=x.device) > mask_ratio).float()
    mask = mask.unsqueeze(-1)  # [seq_len, batch_size, 1]
    return x * mask

def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, optimizer=None, train=False, first_stage=True, mark='train'):
    weight = []
    preds, preds_a, preds_t, preds_v, masks, labels = [], [], [], [], [], []
    loss, losses1, losses2, losses3 = [], [], [], []
    preds_test_condition = []
    dataset = args.dataset
    cuda = torch.cuda.is_available() and not args.no_cuda

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data_idx, data in enumerate(dataloader):
        vidnames = []
        if train: optimizer.zero_grad()
        
        ## read dataloader and generate all missing conditions
        """
        audio_host, text_host, visual_host: [seqlen, batch, dim]
        audio_guest, text_guest, visual_guest: [seqlen, batch, dim]
        qmask: speakers, [batch, seqlen]
        umask: has utt, [batch, seqlen]
        label: [batch, seqlen]
        """
        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]



        qmask, umask, label = data[6], data[7], data[8]
        vidnames += data[-1]
        seqlen = audio_host.size(0)
        batch = audio_host.size(1)

        ## using cmp-net masking manner [at least one view exists]
        ## host mask
        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage) # [seqlen*batch, view_num]
        audio_host_mask = np.reshape(matrix[0], (batch, seqlen, 1))
        text_host_mask = np.reshape(matrix[1], (batch, seqlen, 1))
        visual_host_mask = np.reshape(matrix[2], (batch, seqlen, 1))
        audio_host_mask = torch.LongTensor(audio_host_mask.transpose(1, 0, 2))
        text_host_mask = torch.LongTensor(text_host_mask.transpose(1, 0, 2))
        visual_host_mask = torch.LongTensor(visual_host_mask.transpose(1, 0, 2))
        # guest mask
        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage) # [seqlen*batch, view_num]
        audio_guest_mask = np.reshape(matrix[0], (batch, seqlen, 1))
        text_guest_mask = np.reshape(matrix[1], (batch, seqlen, 1))
        visual_guest_mask = np.reshape(matrix[2], (batch, seqlen, 1))
        audio_guest_mask = torch.LongTensor(audio_guest_mask.transpose(1, 0, 2))
        text_guest_mask = torch.LongTensor(text_guest_mask.transpose(1, 0, 2))
        visual_guest_mask = torch.LongTensor(visual_guest_mask.transpose(1, 0, 2))

        masked_audio_host = audio_host * audio_host_mask
        masked_audio_guest = audio_guest * audio_guest_mask
        masked_text_host = text_host * text_host_mask
        masked_text_guest = text_guest * text_guest_mask
        masked_visual_host = visual_host * visual_host_mask
        masked_visual_guest = visual_guest * visual_guest_mask

        ## add cuda for tensor
        if cuda:
            masked_audio_host, audio_host_mask = masked_audio_host.to(device), audio_host_mask.to(device)
            masked_text_host, text_host_mask = masked_text_host.to(device), text_host_mask.to(device)
            masked_visual_host, visual_host_mask = masked_visual_host.to(device), visual_host_mask.to(device)
            masked_audio_guest, audio_guest_mask = masked_audio_guest.to(device), audio_guest_mask.to(device)
            masked_text_guest, text_guest_mask = masked_text_guest.to(device), text_guest_mask.to(device)
            masked_visual_guest, visual_guest_mask = masked_visual_guest.to(device), visual_guest_mask.to(device)
            # # === 仅在测试阶段加噪声，用于鲁棒性分析 ===
            # if args.add_noise and not train:
            #     noise_level = args.noise_std

            #     masked_audio_host  = masked_audio_host  + noise_level * torch.randn_like(masked_audio_host)
            #     masked_text_host   = masked_text_host   + noise_level * torch.randn_like(masked_text_host)
            #     masked_visual_host = masked_visual_host + noise_level * torch.randn_like(masked_visual_host)

            #     masked_audio_guest  = masked_audio_guest  + noise_level * torch.randn_like(masked_audio_guest)
            #     masked_text_guest   = masked_text_guest   + noise_level * torch.randn_like(masked_text_guest)
            #     masked_visual_guest = masked_visual_guest + noise_level * torch.randn_like(masked_visual_guest)

            qmask = qmask.to(device)
            umask = umask.to(device)
            label = label.to(device)

        ## generate mask_input_features: ? * [seqlen, batch, dim], input_features_mask: ? * [seq_len, batch, 3]
        masked_input_features = generate_inputs(masked_audio_host, masked_text_host, masked_visual_host, \
                                                masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)
        input_features_mask = generate_inputs(audio_host_mask, text_host_mask, visual_host_mask, \
                                                audio_guest_mask, text_guest_mask, visual_guest_mask, qmask)
        mask_a, mask_t, mask_v = input_features_mask[0][:,:,0].transpose(0,1), input_features_mask[0][:,:,1].transpose(0,1), input_features_mask[0][:,:,2].transpose(0,1)

       


        '''
        # masked_input_features, input_features_mask: ?*[seqlen, batch, dim]
        # qmask: speakers, [batch, seqlen]
        # umask: has utt, [batch, seqlen]
        # label: [batch, seqlen]
        # log_prob: [seqlen, batch, num_classes]
        '''

        if args.add_noise and not train:
            masked_input_features[0] = apply_random_mask(masked_input_features[0], mask_ratio=args.noise_std)
        ## forward
        hidden, out, out_a_low,out_a_high, out_t_low,out_t_high, out_v_low,out_v_high,out_final_a,out_final_t,out_final_v,shared_predict, weight_low_save, weight_high_save, proj_a, proj_t, proj_v, shared_a, private_a, shared_t, private_t, shared_v, private_v, feat_a_rec, feat_t_rec, feat_v_rec,private_a_new,private_v_new = model(masked_input_features[0], input_features_mask[0], umask, first_stage)
        if not train and data_idx == 0:
            save_dir = './feature_saves'
            os.makedirs(save_dir, exist_ok=True)
            
            np.save(os.path.join(save_dir, 'out_a_low.npy'), out_a_low.detach().cpu().numpy())
            np.save(os.path.join(save_dir, 'out_a_high.npy'), out_a_high.detach().cpu().numpy())
            np.save(os.path.join(save_dir, 'out_t_low.npy'), out_t_low.detach().cpu().numpy())
            np.save(os.path.join(save_dir, 'out_t_high.npy'), out_t_high.detach().cpu().numpy())
            np.save(os.path.join(save_dir, 'out_v_low.npy'), out_v_low.detach().cpu().numpy())
            np.save(os.path.join(save_dir, 'out_v_high.npy'), out_v_high.detach().cpu().numpy())

        weight_low_save = np.array(weight_low_save)     

        weight_high_save = np.array(weight_low_save)     
        ## save analysis result
        weight.append(weight_low_save)
        weight.append(weight_high_save)
        
        in_mask = torch.clone(input_features_mask[0].permute(1, 0, 2))
        in_mask[umask == 0] = 0
        weight.append(np.array(in_mask.cpu()))
        weight.append(label.detach().cpu().numpy())
        weight.append(vidnames)
        

        ## calculate loss
        lp_ = out.view(-1, out.size(2)) # [batch*seq_len, n_classes]
        # lp_a, lp_t, lp_v = out_a.view(-1, out_a.size(2)), out_t.view(-1, out_t.size(2)), out_v.view(-1, out_v.size(2))

        lp_a_low, lp_t_low, lp_v_low = out_a_low.view(-1, out_a_low.size(2)), out_t_low.view(-1, out_t_low.size(2)), out_v_low.view(-1, out_v_low.size(2))
        lp_a_high, lp_t_high, lp_v_high = out_a_high.view(-1, out_a_high.size(2)), out_t_high.view(-1, out_t_high.size(2)), out_v_high.view(-1, out_v_high.size(2))
        lp_a = lp_a_low + lp_a_high
        lp_t = lp_t_low + lp_t_high
        lp_v = lp_v_low + lp_v_high
        lp_final_a, lp_final_t, lp_final_v = out_final_a.view(-1, out_final_a.size(2)), out_final_t.view(-1, out_final_t.size(2)), out_final_v.view(-1, out_final_v.size(2))
        lp_shared = shared_predict.view(-1, shared_predict.size(2))
        labels_ = label.view(-1) # [batch*seq_len]
        loss_list = []
        if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            if first_stage:
                loss_a = cls_loss(lp_a, labels_, umask)
                loss_t = cls_loss(lp_t, labels_, umask)
                loss_v = cls_loss(lp_v, labels_, umask)
            else:
                loss = cls_loss(lp_, labels_, umask)
                loss = loss_a + loss_t + loss_v
        if dataset in ['CMUMOSI', 'CMUMOSEI']:
            if first_stage:
                loss_a = reg_loss(lp_a, labels_, umask)
                loss_t = reg_loss(lp_t, labels_, umask)
                loss_v = reg_loss(lp_v, labels_, umask)
                loss = loss_a + loss_t + loss_v
            else:
                loss_fused = reg_loss(lp_, labels_, umask)

                # 保证预测都在同一个设备
                lp_ = lp_.to(args.device)
                lp_a = lp_a.to(args.device)
                lp_t = lp_t.to(args.device)
                lp_v = lp_v.to(args.device)
                lp_final_a = lp_final_a.to(args.device)
                lp_final_t = lp_final_t.to(args.device)
                lp_final_v = lp_final_v.to(args.device)
                lp_shared = lp_shared.to(args.device)
                # #  Teacher 引导的一致性 MSE
                # lambda_sem = 0.4  # 可以调高一点
                # sem_loss = semantic_agreement_loss_mse_teacher(lp_, [lp_final_a, lp_final_t, lp_final_v])
                

                # loss = loss_fused + lambda_sem * sem_loss
                lambda_orth = 0.1 #0.1
                loss_orth = (
                    orthogonality_loss(shared_a, private_a) +
                    orthogonality_loss(shared_t, private_t) +
                    orthogonality_loss(shared_v, private_v)
                )

                lambda_recon = 0.2
                recon_loss = (F.mse_loss(feat_a_rec, proj_a.detach()) +
                            F.mse_loss(feat_t_rec, proj_t.detach()) +
                            F.mse_loss(feat_v_rec, proj_v.detach()))

                lambda_private_recon = 0.5  #0.5
                private_recon_loss = (F.mse_loss(private_a_new, private_a.detach())  +
                            F.mse_loss(private_v_new, private_v.detach()))

                lambda_shared_align = 0.4
                loss_shared_align = F.mse_loss(lp_shared, lp_.detach())

                lambda_align = 0.5 #0.5
                cosine_similarity = F.cosine_similarity(shared_a, shared_t, dim=-1)  # [batch, seq_len]
                loss_align_at = 1 - cosine_similarity.mean()

                cosine_similarity = F.cosine_similarity(shared_a, shared_v, dim=-1)
                loss_align_av = 1 - cosine_similarity.mean()

                cosine_similarity = F.cosine_similarity(shared_t, shared_v, dim=-1)
                loss_align_tv = 1 - cosine_similarity.mean()

                loss_align = (loss_align_at + loss_align_av + loss_align_tv) / 3


                loss = loss_fused +  lambda_orth * loss_orth + lambda_recon * recon_loss + lambda_align * loss_align + lambda_private_recon * private_recon_loss + lambda_shared_align * loss_shared_align
                



        if loss is not None:
            loss_list.append(loss)
        else:
            raise RuntimeError("Loss was not computed. Check your dataset and logic conditions.")
        ## save batch results
        preds_a.append(lp_a.data.cpu().numpy())
        preds_t.append(lp_t.data.cpu().numpy())
        preds_v.append(lp_v.data.cpu().numpy())
        preds.append(lp_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        # print(f'---------------{mark} loss: {loss}-------------------')
        preds_test_condition.append(out.view(-1, out.size(2)).data.cpu().numpy())

        if train and first_stage:
            # loss_a.backward()
            # loss_t.backward()
            # loss_v.backward()
            loss_atv = loss_a + loss_t + loss_v
            loss_atv.backward()
            optimizer.step()
        if train and not first_stage:
            loss.backward()
            optimizer.step()

    assert preds!=[], f'Error: no dataset in dataloader'
    preds  = np.concatenate(preds)
    preds_a = np.concatenate(preds_a)
    preds_t = np.concatenate(preds_t)
    preds_v = np.concatenate(preds_v)
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)

    # all
    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        preds = np.argmax(preds, 1)
        preds_a = np.argmax(preds_a, 1)
        preds_t = np.argmax(preds_t, 1)
        preds_v = np.argmax(preds_v, 1)
        avg_loss = round(np.sum(loss)/np.sum(masks), 4)
        avg_accuracy = accuracy_score(labels, preds, sample_weight=masks)
        avg_fscore = f1_score(labels, preds, sample_weight=masks, average='weighted')
        mae = 0
        ua = recall_score(labels, preds, sample_weight=masks, average='macro')
        avg_acc_a = accuracy_score(labels, preds_a, sample_weight=masks)
        avg_acc_t = accuracy_score(labels, preds_t, sample_weight=masks)
        avg_acc_v = accuracy_score(labels, preds_v, sample_weight=masks)
        return mae, ua, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss, weight

    elif dataset in ['CMUMOSI', 'CMUMOSEI']:
        non_zeros = np.array([i for i, e in enumerate(labels) if e != 0]) # remove 0, and remove mask
        # 确保 loss 是 tensor
        if isinstance(loss, list):
            # 将列表中的张量合并
            loss = torch.stack(loss)

        # 如果仍然是 tensor，执行计算
        if isinstance(loss, torch.Tensor):
            avg_loss = round(loss.sum().item() / masks.sum().item(), 4)
        else:
            raise TypeError(f"Unexpected loss type: {type(loss)}")

        

        avg_accuracy = accuracy_score((labels[non_zeros] > 0), (preds[non_zeros] > 0))
        avg_fscore = f1_score((labels[non_zeros] > 0), (preds[non_zeros] > 0), average='weighted')
        mae = np.mean(np.absolute(labels[non_zeros] - preds[non_zeros].squeeze()))
        corr = np.corrcoef(labels[non_zeros], preds[non_zeros].squeeze())[0][1]
        avg_acc_a = accuracy_score((labels[non_zeros] > 0), (preds_a[non_zeros] > 0))
        avg_acc_t = accuracy_score((labels[non_zeros] > 0), (preds_t[non_zeros] > 0))
        avg_acc_v = accuracy_score((labels[non_zeros] > 0), (preds_v[non_zeros] > 0))

        return mae, corr, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss, weight




    # elif dataset in ['CMUMOSI', 'CMUMOSEI']:
    # non_zeros = np.array([i for i, e in enumerate(labels) if e != 0])  # 移除0标签

    # # 确保 loss 是 tensor
    # if isinstance(loss, list):
    #     loss = torch.stack(loss)
    # if isinstance(loss, torch.Tensor):
    #     avg_loss = round(loss.sum().item() / masks.sum().item(), 4)
    # else:
    #     raise TypeError(f"Unexpected loss type: {type(loss)}")

    # ### 原始设置: positive vs negative
    # preds_bin_pos = (preds[non_zeros] > 0)
    # labels_bin_pos = (labels[non_zeros] > 0)
    # avg_accuracy_pos = accuracy_score(labels_bin_pos, preds_bin_pos)
    # avg_fscore_pos = f1_score(labels_bin_pos, preds_bin_pos, average='weighted')

    # ### 新增设置: negative vs non-negative
    # preds_bin_neg = (preds[non_zeros] < 0)
    # labels_bin_neg = (labels[non_zeros] < 0)
    # avg_accuracy_neg = accuracy_score(labels_bin_neg, preds_bin_neg)
    # avg_fscore_neg = f1_score(labels_bin_neg, preds_bin_neg, average='weighted')

    # # 其他指标
    # mae = np.mean(np.absolute(labels[non_zeros] - preds[non_zeros].squeeze()))
    # corr = np.corrcoef(labels[non_zeros], preds[non_zeros].squeeze())[0][1]

    # avg_acc_a = accuracy_score((labels[non_zeros] > 0), (preds_a[non_zeros] > 0))
    # avg_acc_t = accuracy_score((labels[non_zeros] > 0), (preds_t[non_zeros] > 0))
    # avg_acc_v = accuracy_score((labels[non_zeros] > 0), (preds_v[non_zeros] > 0))

    # return {
    #     "mae": mae,
    #     "corr": corr,
    #     "acc_pos": avg_accuracy_pos,
    #     "f1_pos": avg_fscore_pos,
    #     "acc_neg": avg_accuracy_neg,
    #     "f1_neg": avg_fscore_neg,
    #     "acc_atv": [avg_acc_a, avg_acc_t, avg_acc_v],
    #     "vidnames": vidnames,
    #     "loss": avg_loss,
    #     "weight": weight
    # }

  
        




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Params for input
    parser.add_argument('--audio-feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text-feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video-feature', type=str, default=None, help='video feature name')
    parser.add_argument('--dataset', type=str, default='IEMOCAPFour', help='dataset type')

    parser.add_argument('--add-noise', action='store_true', help='whether to add Gaussian noise to features')
    parser.add_argument('--noise-std', type=float, default=0.2, help='standard deviation of Gaussian noise')

    ## Params for model
    parser.add_argument('--time-attn', action='store_true', default=False, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')
    parser.add_argument('--depth', type=int, default=4, help='')
    parser.add_argument('--num_heads', type=int, default=2, help='')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help='')
    parser.add_argument('--hidden', type=int, default=100, help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes [defined by args.dataset]')
    parser.add_argument('--n_speakers', type=int, default=2, help='number of speakers [defined by args.dataset]')

    ## Params for training
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--gpu', type=int, default=2, help='index of gpu')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--num-folder', type=int, default=5, help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--test_condition', type=str, default='atv', choices=['a', 't', 'v', 'at', 'av', 'tv', 'atv'], help='test conditions')
    parser.add_argument('--stage_epoch', type=float, default=100, help='number of epochs of the first stage')




    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    save_folder_name = f'{args.dataset}'
    save_log = os.path.join(config.LOG_DIR, 'main_result', f'{save_folder_name}')
    if not os.path.exists(save_log): os.makedirs(save_log)
    time_dataset = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_{args.dataset}"
    sys.stdout = Logger(filename=f"{save_log}/{time_dataset}_batchsize-{args.batch_size}_lr-{args.lr}_seed-{args.seed}_test-condition-{args.test_condition}.txt",
                        stream=sys.stdout)
    


    ## seed
    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # seed_torch(args.seed)
    # def seed_torch(seed):
        
    #     seed = random.randint(0, 2**32 - 1)
    #     print(f"Using seed: {seed}")
    #     random.seed(seed)
    #     os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化，使得实验可复现
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)  # 多GPU设置
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True

    # 调用函数并设置种子
    seed_torch(args.seed)


    ## dataset
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        args.num_folder = 1
        args.n_classes = 1
        args.n_speakers = 1
    elif args.dataset == 'IEMOCAPFour':
        args.num_folder = 5
        args.n_classes = 4
        args.n_speakers = 2
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder = 5
        args.n_classes = 6
        args.n_speakers = 2
    cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    ## reading data
    print (f'====== Reading Data =======')
    audio_feature, text_feature, video_feature = args.audio_feature, args.text_feature, args.video_feature
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)
    
            
    print(audio_root)
    print(text_root)
    print(video_root)
    
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), f'features not exist!'
    train_loaders, test_loaders, adim, tdim, vdim = get_loaders(audio_root=audio_root,
                                                                             text_root=text_root,
                                                                             video_root=video_root,
                                                                             num_folder=args.num_folder,
                                                                             batch_size=args.batch_size,
                                                                             dataset=args.dataset,
                                                                             num_workers=0)
    assert len(train_loaders) == args.num_folder, f'Error: folder number'

    
    print (f'====== Training and Testing =======')
    folder_mae = []
    folder_corr = []
    folder_acc = []
    folder_f1 = []
    folder_model = []
    for ii in range(args.num_folder):
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        test_loader = test_loaders[ii]
        start_time = time.time()

        print('-'*80)
        print (f'Step1: build model (each folder has its own model)')
        model = build_model(args, adim, tdim, vdim)
        reg_loss = MaskedMSELoss()
        cls_loss = MaskedCELoss()
        if cuda:
            model.to(device)
        optimizer = optim.Adam([{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.l2}])
        print('-'*80)


        print (f'Step2: training (multiple epoches)')
        train_acc_as, train_acc_ts, train_acc_vs = [], [], []
        test_fscores, test_accs, test_maes, test_corrs = [], [], [], []
        models = []
        start_first_stage_time = time.time()

        print("------- Starting the first stage! -------")
        for epoch in range(args.epochs):
            first_stage = True if epoch < args.stage_epoch else False
            ## training and testing (!!! if IEMOCAP, the ua is equal to corr !!!)
            train_mae, train_corr, train_acc, train_fscore, train_acc_atv, train_names, train_loss, weight_train = train_or_eval_model(args, model, reg_loss, cls_loss, train_loader, \
                                                                            optimizer=optimizer, train=True, first_stage=first_stage, mark='train')
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            optimizer=None, train=False, first_stage=first_stage, mark='test')


            ## save
            test_accs.append(test_acc)
            test_fscores.append(test_fscore)
            test_maes.append(test_mae)
            test_corrs.append(test_corr)
            models.append(model)
            train_acc_as.append(train_acc_atv[0])
            train_acc_ts.append(train_acc_atv[1])
            train_acc_vs.append(train_acc_atv[2])

            if first_stage:
                print(f'epoch:{epoch}; a_acc_train:{train_acc_atv[0]:.3f}; t_acc_train:{train_acc_atv[1]:.3f}; v_acc_train:{train_acc_atv[2]:.3f}')
                print(f'epoch:{epoch}; a_acc_test:{train_acc_atv[0]:.3f}; t_acc_test:{train_acc_atv[1]:.3f}; v_acc_test:{train_acc_atv[2]:.3f}')
            else:
                print(f'epoch:{epoch}; train_mae_{args.test_condition}:{train_mae:.3f}; train_corr_{args.test_condition}:{train_corr:.3f}; train_fscore_{args.test_condition}:{train_fscore:2.2%}; train_acc_{args.test_condition}:{train_acc:2.2%}; train_loss_{args.test_condition}:{train_loss}')
                print(f'epoch:{epoch}; test_mae_{args.test_condition}:{test_mae:.3f}; test_corr_{args.test_condition}:{test_corr:.3f}; test_fscore_{args.test_condition}:{test_fscore:2.2%}; test_acc_{args.test_condition}:{test_acc:2.2%}; test_loss_{args.test_condition}:{test_loss}')
            print('-'*10)
            ## update the parameter for the 2nd stage
            if epoch == args.stage_epoch-1:
                model = models[-1]

                model_idx_a = int(torch.argmax(torch.Tensor(train_acc_as)))
                print(f'best_epoch_a: {model_idx_a}')
                model_a = models[model_idx_a]
                transformer_a_para_dict = {k: v for k, v in model_a.state_dict().items() if 'Transformer' in k}
                model.state_dict().update(transformer_a_para_dict)

                model_idx_t = int(torch.argmax(torch.Tensor(train_acc_ts)))
                print(f'best_epoch_t: {model_idx_t}')
                model_t = models[model_idx_t]
                transformer_t_para_dict = {k: v for k, v in model_t.state_dict().items() if 'Transformer' in k}
                model.state_dict().update(transformer_t_para_dict)

                model_idx_v = int(torch.argmax(torch.Tensor(train_acc_vs)))
                print(f'best_epoch_v: {model_idx_v}')
                model_v = models[model_idx_v]
                transformer_v_para_dict = {k: v for k, v in model_v.state_dict().items() if 'Transformer' in k}
                model.state_dict().update(transformer_v_para_dict)

                end_first_stage_time = time.time()
                print("------- Starting the second stage! -------")

        end_second_stage_time = time.time()
        print("-"*80)
        print(f"Time of first stage: {end_first_stage_time - start_first_stage_time}s")
        print(f"Time of second stage: {end_second_stage_time - end_first_stage_time}s")
        print("-" * 80)

        print(f'Step3: saving and testing on the {ii+1} folder')
        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            best_index_test = np.argmax(np.array(test_fscores))
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            best_index_test = np.argmax(np.array(test_accs))


        bestmae = test_maes[best_index_test]
        bestcorr = test_corrs[best_index_test]
        bestf1 = test_fscores[best_index_test]
        bestacc = test_accs[best_index_test]
        bestmodel = models[best_index_test]

        folder_mae.append(bestmae)
        folder_corr.append(bestcorr)
        folder_f1.append(bestf1)
        folder_acc.append(bestacc)
        folder_model.append(bestmodel)
        end_time = time.time()


        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            print(f"The best(acc) epoch of test_condition ({args.test_condition}): {best_index_test} --test_mae {bestmae} --test_corr {bestcorr} --test_fscores {bestf1} --test_acc {bestacc}.")
            print(f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>')
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            print(f"The best(acc) epoch of test_condition ({args.test_condition}): {best_index_test} --test_acc {bestacc} --test_ua {bestcorr}.")
            print(f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>')

    print('-'*80)
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        print(f"Folder avg: test_condition ({args.test_condition}) --test_mae {np.mean(folder_mae)} --test_corr {np.mean(folder_corr)} --test_fscores {np.mean(folder_f1)} --test_acc{np.mean(folder_acc)}")
    if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        print(f"Folder avg: test_condition ({args.test_condition}) --test_acc{np.mean(folder_acc)} --test_ua {np.mean(folder_corr)}")

    print (f'====== Saving =======')
    save_model = os.path.join(config.MODEL_DIR, 'main_result', f'{save_folder_name}')
    if not os.path.exists(save_model): os.makedirs(save_model)
    ## gain suffix_name
    suffix_name = f"{time_dataset}_hidden-{args.hidden}_bs-{args.batch_size}"
    ## gain feature_name
    feature_name = f'{audio_feature};{text_feature};{video_feature}'
    ## gain res_name
    mean_mae = np.mean(np.array(folder_mae))
    mean_corr = np.mean(np.array(folder_corr))
    mean_f1 = np.mean(np.array(folder_f1))
    mean_acc = np.mean(np.array(folder_acc))
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        res_name = f'mae-{mean_mae:.3f}_corr-{mean_corr:.3f}_f1-{mean_f1:.4f}_acc-{mean_acc:.4f}'
    if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        res_name = f'acc-{mean_acc:.4f}_ua-{mean_corr:.4f}'
    save_path = f'{save_model}/{suffix_name}_features-{feature_name}_{res_name}_test-condition-{args.test_condition}.pth'
    torch.save({'model': model.state_dict()}, save_path)
    print(save_path)
