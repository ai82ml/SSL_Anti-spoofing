import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from transformers import AutoProcessor, ASTModel
from torch.utils.data import DataLoader, Subset
import yaml
from data_utils_SSL import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from model import Model_wav2vec, AST_Vector_not, AST_Vector_160_not, CombinedClassifier_1layer_add, CombinedClassifier_2layer_add
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed


def evaluate_accuracy(dev_loader, model, device, processor):
    val_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in dev_loader:        
        batch_size = batch_x.size(0)
        num_total += batch_size    
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        #batch_out = model(batch_x)
        #batch_x_ast = processor(batch_x.numpy(), return_tensors="pt", sampling_rate=16000).to(device)
        batch_x_ast = processor(batch_x.numpy(), return_tensors="pt", sampling_rate=16000)
        batch_x_ast = batch_x_ast["input_values"].to(device)
        
        batch_x_wav2vec = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        #batch_out_ast = ast_model(**batch_x_ast)
        #batch_out_wav2vec = wav2vec_model(batch_x_wav2vec)
        batch_out = mix_model(wav2vec_input=batch_x_wav2vec, ast_input=batch_x_ast)
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        
    val_loss /= num_total
    valid_accuracy = 100 * (num_correct / num_total)
    return val_loss, valid_accuracy


def produce_evaluation_file(dataset, mix_model, device, save_path, processor):
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    mix_model.eval()
    
    fname_list = []
    key_list = []
    score_list = []

    for batch_x,utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x_ast = processor(batch_x.numpy(), return_tensors="pt", sampling_rate=16000)
        batch_x_ast = batch_x_ast["input_values"].to(device)
        batch_x_wav2vec = batch_x.to(device)
        
        batch_out = mix_model(wav2vec_input=batch_x_wav2vec, ast_input=batch_x_ast)
        
        batch_score = (batch_out[:, 1]  
                       ).data.cpu().numpy().ravel() 
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, lr,optim, device, processor):
    running_loss = 0    
    num_total = 0.0
    num_correct = 0.0
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in train_loader:      
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        #batch_x_ast = processor(batch_x.numpy(), return_tensors="pt", sampling_rate=16000).to(device)
        batch_x_ast = processor(batch_x.numpy(), return_tensors="pt", sampling_rate=16000)
        batch_x_ast = batch_x_ast["input_values"].to(device)
        
        batch_x_wav2vec = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        #batch_out_ast = ast_model(**batch_x_ast)
        #batch_out_wav2vec = wav2vec_model(batch_x_wav2vec)
        batch_out = mix_model(wav2vec_input=batch_x_wav2vec, ast_input=batch_x_ast)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += (batch_loss.item() * batch_size)
        
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/home/audio/ASVspoof2019/LA/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
    '''
    parser.add_argument('--protocols_path', type=str, default='/home/audio/ASVspoof2019/LA/', help='Change with path to user\'s LA database protocols directory address')
    '''
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
    '''

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='LA',choices=['LA', 'PA','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 


    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    track = args.track

    assert track in ['LA', 'PA','DF'], 'Invalid track given'

    #database
    prefix      = 'ASVspoof2019_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    ast_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", output_hidden_states=True)
    ast_model_vector160 = AST_Vector_160_not(ast_model, embed_size=768)
    #ast_model_vector160 =ast_model_vector160.to(device)
    
    wav2vec_model = Model_wav2vec(args,device)

    mix_model = CombinedClassifier_1layer_add(ast_model_vector160, wav2vec_model)
    #mix_model = CombinedClassifier(ast_model_vector160, wav2vec_model)
    mix_model =mix_model.to(device)
    #print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(mix_model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if args.model_path:
        mix_model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    #evaluation 
    if args.eval:
        file_eval = genSpoof_list(dir_meta = '/home/audio/ASVspoof2021/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt',is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join('/home/audio/ASVspoof2021/'+'ASVspoof2021_{}_eval/'.format(args.track)))
        #eval_set= Subset(eval_set, list(range(20)))
        #start_index = 82573
        #indices = list(range(start_index, len(eval_set)))

        # サブセットを作成
        #eval_set= Subset(eval_set, indices)
        produce_evaluation_file(eval_set, mix_model, device, args.eval_output, processor)
        sys.exit(0)
     
    # define train dataloader
    d_label_trn,file_train = genSpoof_list(dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix,prefix_2019)),is_train=True,is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'{}_{}_train/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
    
    num_samples = 34
    #train_set = Subset(train_set, list(range(num_samples)))
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define validation dataloader
    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix,prefix_2019)),is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,
		labels = d_label_dev,
		base_dir = os.path.join(args.database_path+'{}_{}_dev/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
    
    num_samples1 = 35
    #dev_set = Subset(dev_set, list(range(num_samples1)))
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    del dev_set,d_label_dev
    

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_loss = 1
    for epoch in range(num_epochs):        
        running_loss, train_accuracy = train_epoch(train_loader, mix_model, args.lr,optimizer, device,processor)
        val_loss, valid_accuracy = evaluate_accuracy(dev_loader, mix_model, device,processor)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - running_loss:{} - train_accuracy:{} - valid_loss:{} - valid_accuracy:{}'.format(epoch, running_loss, train_accuracy, val_loss, valid_accuracy))
        if val_loss < best_loss:
            best_loss = val_loss
            print('best model find at epoch', epoch)
        torch.save(mix_model.state_dict(), os.path.join(
            '/home/haruto/SSL_Anti-spoofing/models/ast-wav2vec2-長さ調整あり-パラメータ固定-正しい-ドロップアウトなし全結合層追加', 'epoch_{}.pth'.format(epoch)))