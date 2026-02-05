import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse

import warnings

from models.model import Encoder, Classifier, Decoder
from utils.data_utils import get_dataset
from utils.loss import small_loss_selection
from utils.pseudo_label_utils import generate_pseudo_labels
from utils.utils import set_seed, new_length, downsample_torch, get_accuracy, count_refurb_matrix, f1_scores, pairflip_penalty, create_dir

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# Base setup
parser.add_argument('--random_seed', type=int, default=42, help='seed')
parser.add_argument('--data_dir', type=str, default='./dataset_dir/', help='dataset directory')
parser.add_argument('--model_save_dir', type=str, default='./outputs/model_save/', help='model save directory')
parser.add_argument('--result_save_dir', type=str, default='./outputs/result_save/', help='result save directory')
parser.add_argument('--loss_save_dir', type=str, default='./outputs/loss_save/', help='result save directory')
parser.add_argument('--model_names', type=list, default=['encoder1.pth', 'encoder2.pth', 'encoder3.pth'], help='model names')

# Dataset setup
parser.add_argument('--archive', type=str, default='UEA', help='UEA, other')
parser.add_argument('--dataset', type=str, default='ArticularyWordRecognition', help='dataset name')  # [all dataset in Multivariate_arff]

# Label noise
parser.add_argument('--noise_type', type=str, default='symmetric', help='symmetric, instance, pairflip')
parser.add_argument('--label_noise_rate', type=float, default=0.2, help='label noise ratio')
parser.add_argument('--scale_list', type=list, default=[1, 2, 4], help='')

# training setup
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='')
parser.add_argument('--epoch', type=int, default=50, help='training epoch')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers')

# model setup
parser.add_argument('--embedding_size', type=int, default=16, help='model hyperparameters')
parser.add_argument('--feature_size', type=int, default=64, help='model output dimension')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
parser.add_argument('--num_heads', type=int, default=4, help='')

# feature augmentation
parser.add_argument('--alpha', type=float, default=0.01, help='the hyperparameter alpha of MixDecomposition')
parser.add_argument('--beta', type=float, default=0.5, help='the hyperparameter beta of MixDecomposition')

# delay loss and refurb setup
parser.add_argument('--start_mask_epoch', type=int, default=10, help='using small loss criterion epoch')
parser.add_argument('--start_refurb', type=int, default=30, help='start refurb epoch')
parser.add_argument('--refurb_len', type=int, default=5, help='the length of refurb')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')

# GPU setup
parser.add_argument('--gpu_id', type=int, default=0)

args = parser.parse_args()

def pretrain(args, train_loader, encoders, decoder, model_path, pretrain_optimizer):
    # train encoder and decoder
    encoder1, encoder2, encoder3 = encoders[0], encoders[1], encoders[2]
    mse_criterion = nn.MSELoss().cuda()

    best_loss = float('inf')
    for epoch in range(args.epoch):
        encoder1.train()
        encoder2.train()
        encoder3.train()
        decoder.train()

        epoch_train_loss = 0
        for i, (x, y, _, indexes) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()

            # downsample
            input1 = downsample_torch(x, sample_rate=args.scale_list[0]).transpose(1, 2)
            input2 = downsample_torch(x, sample_rate=args.scale_list[1]).transpose(1, 2)
            input3 = downsample_torch(x, sample_rate=args.scale_list[2]).transpose(1, 2)
            x = x.transpose(1, 2)

            encoder_output1 = encoder1(input1)
            encoder_output2 = encoder2(input2)
            encoder_output3 = encoder3(input3)

            decoder_input = torch.cat([encoder_output1, encoder_output2, encoder_output3], dim=1)
            decoder_output = decoder(decoder_input)
            loss1 = mse_criterion(decoder_output, x)

            pretrain_optimizer.zero_grad()
            loss1.backward()
            pretrain_optimizer.step()

            epoch_train_loss += loss1.item() * x.shape[0]

        epoch_train_loss = epoch_train_loss / len(train_loader.dataset)
        print('Epoch:', epoch, 'train Loss:', epoch_train_loss)

        # save best model
        if epoch_train_loss < best_loss:
            best_loss = epoch_train_loss
            torch.save(encoder1.state_dict(), model_path + args.model_names[0])
            torch.save(encoder2.state_dict(), model_path + args.model_names[1])
            torch.save(encoder3.state_dict(), model_path + args.model_names[2])

def evaluate(args, epoch, test_loader, encoders, classifiers):
    encoder1, encoder2, encoder3 = encoders[0], encoders[1], encoders[2]
    classifier1, classifier2, classifier3 = classifiers[0], classifiers[1], classifiers[2]

    encoder1.eval()
    encoder2.eval()
    encoder3.eval()
    classifier1.eval()
    classifier2.eval()
    classifier3.eval()

    test_correct, epoch_test_loss, f1_macro, f1_weighted, f1_micro = 0, 0, 0, 0, 0
    test_num = 0
    for i, (x, y, indexes) in enumerate(test_loader):
        test_num += 1
        with torch.no_grad():
            x, y = x.cuda(), y.cuda()

            input1 = downsample_torch(x, sample_rate=args.scale_list[0]).transpose(1, 2)
            input2 = downsample_torch(x, sample_rate=args.scale_list[1]).transpose(1, 2)
            input3 = downsample_torch(x, sample_rate=args.scale_list[2]).transpose(1, 2)

            # foreward
            encoder_output1 = encoder1(input1)
            encoder_output2 = encoder2(input2)
            encoder_output3 = encoder3(input3)
            c_out1 = classifier1(torch.cat([encoder_output1, encoder_output2], dim=1))
            c_out2 = classifier2(torch.cat([encoder_output1, encoder_output3], dim=1))
            c_out3 = classifier3(torch.cat([encoder_output2, encoder_output3], dim=1))

            loss1 = F.cross_entropy(c_out1, y, reduction='mean')
            loss2 = F.cross_entropy(c_out2, y, reduction='mean')
            loss3 = F.cross_entropy(c_out3, y, reduction='mean')

            loss = (loss1 + loss2 + loss3)
            epoch_test_loss += loss.item() * int(x.shape[0])
            test_cor = get_accuracy(c_out1, c_out2, c_out3, y)
            test_correct += test_cor

            f1_mac, f1_w, f1_mic = f1_scores(c_out1, c_out2, c_out3, y)
            f1_macro += f1_mac
            f1_weighted += f1_w
            f1_micro += f1_mic

    epoch_test_loss = epoch_test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)
    f1_macro = f1_macro / test_num
    f1_micro = f1_micro / test_num
    f1_weighted = f1_weighted / test_num

    return epoch_test_loss, test_acc, f1_macro, f1_weighted, f1_micro

def train(args, epoch, train_loader, train_dataset, train_target, encoders, classifiers, optimizer, refurb_matrixs, unselected_inds, update_inds):
    encoder1, encoder2, encoder3 = encoders[0], encoders[1], encoders[2]
    classifier1, classifier2, classifier3 = classifiers[0], classifiers[1], classifiers[2]
    refurb_matrix1, refurb_matrix2, refurb_matrix3 = refurb_matrixs[0], refurb_matrixs[1], refurb_matrixs[2]

    encoder1.eval()
    encoder2.eval()
    encoder3.eval()
    classifier1.train()
    classifier2.train()
    classifier3.train()

    remember_rate = 1 if epoch < args.start_mask_epoch else 1 - args.label_noise_rate
    epoch_train_loss, epoch_train_correct = 0, 0
    loss_all = np.zeros(len(train_dataset))

    train_num = 0
    for i, (x, y, _, indexes) in enumerate(train_loader):
        train_num += 1
        x, y = x.cuda(), y.cuda()

        # downsample
        input1 = downsample_torch(x, sample_rate=args.scale_list[0]).transpose(1, 2)
        input2 = downsample_torch(x, sample_rate=args.scale_list[1]).transpose(1, 2)
        input3 = downsample_torch(x, sample_rate=args.scale_list[2]).transpose(1, 2)

        # foreward
        encoder_output1 = encoder1(input1)
        encoder_output2 = encoder2(input2)
        encoder_output3 = encoder3(input3)
        c_out1 = classifier1(torch.cat([encoder_output1, encoder_output2], dim=1))
        c_out2 = classifier2(torch.cat([encoder_output1, encoder_output3], dim=1))
        c_out3 = classifier3(torch.cat([encoder_output2, encoder_output3], dim=1))

        # compute refurb matrix
        refurb_matrix1, refurb_matrix2, refurb_matrix3 = count_refurb_matrix(c_out1, c_out2, c_out3, [refurb_matrix1, refurb_matrix2, refurb_matrix3], args.refurb_len, indexes, epoch)

        pseudo_labels, update_inds = generate_pseudo_labels(args, epoch, y, [refurb_matrix1, refurb_matrix2, refurb_matrix3], indexes, unselected_inds)

        # compute delay loss
        loss1, loss2, loss3, loss_all = small_loss_selection(args, [c_out1, c_out2, c_out3], pseudo_labels, epoch, loss_all, indexes, update_inds)

        if args.noise_type == 'pairflip':
            penalty1 = pairflip_penalty(args, c_out1)
            penalty2 = pairflip_penalty(args, c_out2)
            penalty3 = pairflip_penalty(args, c_out3)
            loss1 += penalty1
            loss2 += penalty2
            loss3 += penalty3

        optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward()
        optimizer.step()

        loss = (loss1 + loss3 + loss3) / 3
        epoch_train_loss += loss.item() * int(x.shape[0] * remember_rate)
        test_cor = get_accuracy(c_out1, c_out2, c_out3, y)
        epoch_train_correct += test_cor

    epoch_train_loss = epoch_train_loss / int(len(train_dataset) * remember_rate)
    epoch_train_acc = epoch_train_correct / len(train_dataset)

    # obtain unselected inds
    if epoch == args.start_mask_epoch:
        ind_1_sorted = torch.argsort(torch.from_numpy(loss_all).cuda(), descending=True)
        for i in range(int(len(ind_1_sorted) * args.label_noise_rate)):
            unselected_inds.append(ind_1_sorted[i].cpu().numpy().item())

    return epoch_train_loss, epoch_train_acc, train_loader, train_target, [encoder1, encoder2, encoder3], [classifier1, classifier2, classifier3], [refurb_matrix1, refurb_matrix2, refurb_matrix3], unselected_inds, update_inds


def main():
    set_seed(args)

    # get dataset
    train_loader, test_loader, input_dimension, seq_length, num_classes = get_dataset(args)
    train_target = train_loader.dataset.target
    args.input_dimension, args.seq_length, args.num_classes = input_dimension, seq_length, num_classes

    # building model
    encoder1 = Encoder(new_length(seq_length, args.scale_list[0]), input_dimension, args.embedding_size, args.feature_size, args.num_layers, args.num_heads).cuda()
    encoder2 = Encoder(new_length(seq_length, args.scale_list[1]), input_dimension, args.embedding_size, args.feature_size, args.num_layers, args.num_heads).cuda()
    encoder3 = Encoder(new_length(seq_length, args.scale_list[2]), input_dimension, args.embedding_size, args.feature_size, args.num_layers, args.num_heads).cuda()
    decoder = Decoder(args.feature_size * len(args.scale_list), seq_length, input_dimension).cuda()
    classifier1 = Classifier(args.feature_size * 2, num_classes).cuda()
    classifier2 = Classifier(args.feature_size * 2, num_classes).cuda()
    classifier3 = Classifier(args.feature_size * 2, num_classes).cuda()

    # define refurb matrix
    refurb_matrix1, refurb_matrix2, refurb_matrix3 = (np.zeros((len(train_loader.dataset), args.refurb_len, num_classes)),
                                                      np.zeros((len(train_loader.dataset), args.refurb_len, num_classes)),
                                                      np.zeros((len(train_loader.dataset), args.refurb_len, num_classes)))

    # define optimizer
    optimizer = torch.optim.Adam([{'params': encoder1.parameters()}, {'params': encoder2.parameters()}, {'params': encoder3.parameters()},
                                 {'params': classifier1.parameters()}, {'params': classifier2.parameters()}, {'params': classifier3.parameters()},
                                 {'params': decoder.parameters()}], lr=args.lr)

    # define start epoch and best loss
    unselected_inds, update_inds = [], []
    last_five_accs, last_five_losses, last_five_f1_macro, last_five_f1_weighted, last_five_f1_micro = [], [], [], [], []

    # pretrain model if model parameters do not exist
    create_dir(args.model_save_dir)
    if not os.path.exists(args.model_save_dir + '/' + args.model_names[0]):
        # if not exist, pretrain and save model
        pretrain(args, train_loader, [encoder1, encoder2, encoder3], decoder, args.model_save_dir, optimizer)

    # load pretrain model
    encoder1.load_state_dict(torch.load(args.model_save_dir + args.model_names[0]))
    encoder2.load_state_dict(torch.load(args.model_save_dir + args.model_names[1]))
    encoder3.load_state_dict(torch.load(args.model_save_dir + args.model_names[2]))

    for epoch in range(args.epoch):
        # train
        train_loss, train_acc, train_loader, train_target, encoders, classifiers, refurb_matrixs, unselected_inds, update_inds = (
            train(args, epoch, train_loader, train_loader.dataset.dataset, train_target, [encoder1, encoder2, encoder3],[classifier1, classifier2, classifier3], optimizer,
                  [refurb_matrix1, refurb_matrix2, refurb_matrix3], unselected_inds, update_inds))

        encoder1, encoder2, encoder3 = encoders[0], encoders[1], encoders[2]
        classifier1, classifier2, classifier3 = classifiers[0], classifiers[1], classifiers[2]
        refurb_matrix1, refurb_matrix2, refurb_matrix3 = refurb_matrixs[0], refurb_matrixs[1], refurb_matrixs[2]

        # test
        test_loss, test_acc, f1_macro, f1_weighted, f1_micro = evaluate(args, epoch, test_loader, encoders, classifiers)
        print('Epoch:[%d/%d] train_loss:%f, train_acc:%f, test_loss:%f, test_acc:%f' % (epoch + 1, args.epoch, train_loss, train_acc, test_loss, test_acc))

        # compute last five accs and losses
        if (epoch + 5) >= args.epoch:
            last_five_losses.append(test_loss)
            last_five_accs.append(test_acc)
            last_five_f1_macro.append(f1_macro)
            last_five_f1_weighted.append(f1_weighted)
            last_five_f1_micro.append(f1_micro)

    test_accuracy = round(np.mean(last_five_accs), 4)
    test_loss = round(np.mean(last_five_losses), 4)
    f1_macro = round(np.mean(last_five_f1_macro), 4)
    f1_weighted = round(np.mean(last_five_f1_weighted), 4)
    f1_micro = round(np.mean(last_five_f1_micro), 4)
    print('Test Accuracy:', test_accuracy, 'Test Loss:', test_loss, 'F1_macro:', f1_macro, 'F1_weighted:', f1_weighted, 'F1_micro:', f1_micro)

if __name__ == '__main__':
    main()
  

