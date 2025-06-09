import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse

import warnings

from models.model import Encoder, Classifier, Decoder
from utils.data_utils import get_dataset, load_loader
from utils.loss import Discrimn_Loss, delay_loss
from utils.pseudo_label_utils import generate_pseudo_labels
from utils.utils import set_seed, new_length, downsample_torch, get_accuracy, create_dir, create_file, refurb_label, \
    count_refurb_matrix, f1_scores, plot_tsne

from utils.constants import Multivariate2018_arff_DATASET as UEA_DATASET
from utils.constants import Four_dataset as OTHER_DATASET

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# Base setup
parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')
parser.add_argument('--data_dir', type=str, default='../data/Multivariate2018_arff/Multivariate_arff', help='dataset directory')
parser.add_argument('--model_save_dir', type=str, default='./outputs/model_save/', help='model save directory')
parser.add_argument('--result_save_dir', type=str, default='./outputs/result_save/', help='result save directory')
parser.add_argument('--loss_save_dir', type=str, default='./outputs/loss_save/', help='result save directory')
parser.add_argument('--model_names', type=list, default=['encoder1.pth', 'encoder2.pth', 'encoder3.pth'], help='model names')

# Dataset setup
parser.add_argument('--archive', type=str, default='UEA', help='UEA, other')
parser.add_argument('--dataset', type=str, default='ArticularyWordRecognition', help='dataset name')  # [all dataset in Multivariate_arff]

# Label noise
parser.add_argument('--noise_type', type=str, default='symmetric', help='symmetric, instance, pairflip')
parser.add_argument('--label_noise_rate', type=float, default=0.2, help='label noise ratio, sym: 0.2, 0.5, asym: 0.4, ins: 0.4')
parser.add_argument('--scale_list', type=list, default=[1, 2, 4], help='')

# training setup
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='')
parser.add_argument('--epoch', type=int, default=200, help='training epoch')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers')

# model setup
parser.add_argument('--embedding_size', type=int, default=16, help='model hyperparameters')
parser.add_argument('--feature_size', type=int, default=64, help='model output dimension')
parser.add_argument('--num_layers', type=int, default=2, help='number of  layers')
parser.add_argument('--num_heads', type=int, default=4, help='')

# feature augmentation and MCR setup
parser.add_argument('--alpha', type=float, default=-0.01, help='MCR hyperparameters')
parser.add_argument('--beta', type=float, default=0.01, help='feature augmentation hyperparameters')
parser.add_argument('--gamma', type=float, default=0.1, help='feature augmentation hyperparameters')

# delay loss and refurb setup
parser.add_argument('--start_mask_epoch', type=int, default=10, help='sample mask epoch')
parser.add_argument('--start_delay_loss', type=int, default=15, help='start delayed loss epoch')
parser.add_argument('--delay_loss_k', type=int, default=3, help='the length of delay loss')
parser.add_argument('--start_refurb', type=int, default=1000, help='start refurb epoch')
parser.add_argument('--refurb_len', type=int, default=5, help='the length of refurb')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')

# GPU setup
parser.add_argument('--gpu_id', type=int, default=0)

args = parser.parse_args()

def pretrain(args, train_loader, encoders, decoder, model_path, pretrain_optimizer):
    out_dir = args.loss_save_dir + args.archive + '/' + str(args.dataset) + '/'
    loss_file = 'loss_%s_%s%.2f_seed%d.txt' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed)
    loss_file = create_file(out_dir, loss_file, 'epoch,train loss')

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
        for i, (x, y, indexes) in enumerate(train_loader):
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
        with open(loss_file, "a") as myfile:
            myfile.write(str('Epoch:[%d/%d] train_loss:%.4f\n' % (epoch + 1, args.epoch, epoch_train_loss)))

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

    model_out = np.zeros((len(test_loader.dataset), args.num_classes))
    y_pred = np.zeros(len(test_loader.dataset))

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
            test_cor, target_pred = get_accuracy(c_out1, c_out2, c_out3, y)
            test_correct += test_cor

            f1_mac, f1_w, f1_mic = f1_scores(c_out1, c_out2, c_out3, y)
            f1_macro += f1_mac
            f1_weighted += f1_w
            f1_micro += f1_mic

            model_out[indexes] = c_out1.detach().cpu().numpy()
            y_pred[indexes] = target_pred.detach().cpu().numpy()

    epoch_test_loss = epoch_test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)
    f1_macro = f1_macro / test_num
    f1_micro = f1_micro / test_num
    f1_weighted = f1_weighted / test_num

    # if epoch == args.epoch - 1:
    #     plot_tsne(model_out, y_pred, '_')

    return epoch_test_loss, test_acc, f1_macro, f1_weighted, f1_micro

def train(epoch, train_loader, train_dataset, train_target, encoders, classifiers, optimizer, refurb_matrixs, unselected_inds, update_inds):
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

    for i, (x, y, indexes) in enumerate(train_loader):
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
        refurb_matrix1, refurb_matrix2, refurb_matrix3 = (
            count_refurb_matrix(c_out1, c_out2, c_out3, [refurb_matrix1, refurb_matrix2, refurb_matrix3], args.refurb_len, indexes, epoch))

        pseudo_labels = generate_pseudo_labels(args, epoch, y, [refurb_matrix1, refurb_matrix2, refurb_matrix3], indexes, unselected_inds)

        # compute delay loss
        loss1, loss2, loss3, loss_all = delay_loss(args, [c_out1, c_out2, c_out3], pseudo_labels, epoch, loss_all, indexes, update_inds)

        optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward()
        optimizer.step()

        loss = (loss1 + loss3 + loss3) / 3
        epoch_train_loss += loss.item() * int(x.shape[0] * remember_rate)
        test_cor, target_pred = get_accuracy(c_out1, c_out2, c_out3, y)
        epoch_train_correct += test_cor

    epoch_train_loss = epoch_train_loss / int(len(train_dataset) * remember_rate)
    epoch_train_acc = epoch_train_correct / len(train_dataset)

    # obtain unselected inds
    if epoch == args.start_mask_epoch:
        ind_1_sorted = torch.argsort(torch.from_numpy(loss_all).cuda(), descending=True)
        for i in range(int(len(ind_1_sorted) * args.label_noise_rate)):
            unselected_inds.append(ind_1_sorted[i].cpu().numpy().item())

    return epoch_train_loss, epoch_train_acc, train_loader, train_target, [encoder1, encoder2, encoder3], [classifier1, classifier2, classifier3], [refurb_matrix1, refurb_matrix2, refurb_matrix3], unselected_inds, update_inds, loss_all


def main(archive='UEA', gpu_id=0, noise_type='symmetric', noise_rates=[0.5], start_refurbs=[30], refurb_lens=[5], use_tsda=True, lams=[0.4]):
    if archive == 'UEA':
        args.archive = archive
        datasets = UEA_DATASET
        args.data_dir = '../data/Multivariate2018_arff/Multivariate_arff/'
    elif archive == 'other':
        args.archive = archive
        datasets = OTHER_DATASET
        args.data_dir = '../data/ts_noise_data/'
    torch.cuda.set_device(gpu_id)

    seeds = [256]
    # seeds = [256, 128, 96, 42, 40]

    for start_refurb in start_refurbs:
        for refurb_len in refurb_lens:
            args.start_refurb = start_refurb
            args.refurb_len = refurb_len
            for lam in lams:
                args.lam = lam

                for dataset in datasets:
                    args.dataset = dataset
                    out_dir = args.result_save_dir + args.archive + '/' + str(args.dataset) + '/'
                    total_file = create_file(out_dir, 'total_result.txt', 'state,test_acc_list,mean_test_acc,f1_macro_list,f1_macro,f1_weighted_list,f1_weighted,f1_micro_list,f1_micro', exist_create_flag=False)

                    for noise_rate in noise_rates:
                        args.noise_type = noise_type
                        args.label_noise_rate = noise_rate

                        test_acc_list, f1_macro_list, f1_weighted_list, f1_micro_list = [], [], [], []
                        for seed in seeds:
                            args.random_seed = seed
                            set_seed(args)

                            # get dataset
                            train_loader, test_loader, train_dataset, train_target, test_dataset, test_target, num_classes, clean_inds, noise_inds = get_dataset(args, use_tsda)
                            args.num_classes = num_classes

                            input_dimension = train_dataset.shape[1]  # input feature dimension
                            seq_length = train_dataset.shape[2]  # input sequence length

                            # load model
                            encoder1 = Encoder(new_length(seq_length, args.scale_list[0]), input_dimension, args.embedding_size, args.feature_size, args.num_layers, args.num_heads).cuda()
                            encoder2 = Encoder(new_length(seq_length, args.scale_list[1]), input_dimension, args.embedding_size, args.feature_size, args.num_layers, args.num_heads).cuda()
                            encoder3 = Encoder(new_length(seq_length, args.scale_list[2]), input_dimension, args.embedding_size, args.feature_size, args.num_layers, args.num_heads).cuda()
                            decoder = Decoder(args.feature_size * len(args.scale_list), seq_length, input_dimension).cuda()

                            classifier1 = Classifier(args.feature_size * 2, num_classes).cuda()
                            classifier2 = Classifier(args.feature_size * 2, num_classes).cuda()
                            classifier3 = Classifier(args.feature_size * 2, num_classes).cuda()

                            # define refurb matrix
                            refurb_matrix1, refurb_matrix2, refurb_matrix3 = (np.zeros((len(train_target), args.refurb_len, num_classes)),
                                                                              np.zeros((len(train_target), args.refurb_len, num_classes)),
                                                                              np.zeros((len(train_target), args.refurb_len, num_classes)))

                            # define optimizer
                            optimizer = torch.optim.Adam([{'params': encoder1.parameters()}, {'params': encoder2.parameters()}, {'params': encoder3.parameters()},
                                                         {'params': classifier1.parameters()}, {'params': classifier2.parameters()}, {'params': classifier3.parameters()},
                                                         {'params': decoder.parameters()}], lr=args.lr)

                            # define start epoch and best loss
                            unselected_inds, update_inds = [], []
                            last_five_accs, last_five_losses, last_five_f1_macro, last_five_f1_weighted, last_five_f1_micro = [], [], [], [], []

                            if use_tsda:
                                args.model_names = ['%s_%s%.2f_seed%d_fomer_1' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed),
                                                    '%s_%s%.2f_seed%d_fomer_2' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed),
                                                    '%s_%s%.2f_seed%d_fomer_3' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed)]
                            else:
                                args.model_names = ['%s_%s%.2f_seed%d_notsda_fomer_1' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed),
                                                     '%s_%s%.2f_seed%d_notsda_fomer_2' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed),
                                                     '%s_%s%.2f_seed%d_notsda_fomer_3' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed)]

                            # pretrain model if model parameters do not exist
                            model_path = args.model_save_dir + args.archive + '/' + str(args.dataset) + '/'
                            create_dir(model_path)
                            if not os.path.exists(args.model_save_dir + args.archive + '/' + str(args.dataset) + '/' + args.model_names[0]):
                                # if not exist, pretrain and save model
                                pretrain(args, train_loader, [encoder1, encoder2, encoder3], decoder, model_path, optimizer)

                            # load pretrain model
                            encoder1.load_state_dict(torch.load(model_path + args.model_names[0]))
                            encoder2.load_state_dict(torch.load(model_path + args.model_names[1]))
                            encoder3.load_state_dict(torch.load(model_path + args.model_names[2]))

                            out_file = '%s_%s%.1f_startrefurb%d_len%d.txt' % (args.dataset, args.noise_type, args.label_noise_rate, args.start_refurb, args.refurb_len)
                            out_file = create_file(out_dir, out_file, 'epoch,train loss,train acc,test loss,test acc')
                            loss_file = create_file(path=out_dir, filename='true_false_loss.txt', write_line='mean_true_loss,std_true_loss,mean_false_loss,std_false_loss')
                            for epoch in range(args.epoch):
                                # train
                                train_loss, train_acc, train_loader, train_target, encoders, classifiers, refurb_matrixs, unselected_inds, update_inds, loss_all = (
                                    train(epoch, train_loader, train_dataset, train_target, [encoder1, encoder2, encoder3],[classifier1, classifier2, classifier3], optimizer,
                                          [refurb_matrix1, refurb_matrix2, refurb_matrix3], unselected_inds, update_inds))

                                encoder1, encoder2, encoder3 = encoders[0], encoders[1], encoders[2]
                                classifier1, classifier2, classifier3 = classifiers[0], classifiers[1], classifiers[2]
                                refurb_matrix1, refurb_matrix2, refurb_matrix3 = refurb_matrixs[0], refurb_matrixs[1], refurb_matrixs[2]

                                # test
                                test_loss, test_acc, f1_macro, f1_weighted, f1_micro = evaluate(args, epoch, test_loader, encoders, classifiers)
                                print('Epoch:[%d/%d] train_loss:%f, train_acc:%f, test_loss:%f, test_acc:%f' % (
                                    epoch + 1, args.epoch, train_loss, train_acc, test_loss, test_acc))

                                with open(out_file, "a") as myfile:
                                    myfile.write(str('Epoch:[%d/%d] train_loss:%f, train_acc:%f, test_loss:%f, test_acc:%f' % (
                                        epoch + 1, args.epoch, train_loss, train_acc, test_loss, test_acc) + "\n"))

                                true_loss = loss_all[clean_inds]
                                false_loss = loss_all[noise_inds]
                                mean_true_loss, std_true_loss = np.mean(true_loss), np.std(true_loss)
                                mean_false_loss, std_false_loss = np.mean(false_loss), np.std(false_loss)
                                with open(loss_file, "a") as myfile:
                                    myfile.write(str('%d,%.4f,%.4f,%.4f,%.4f\n' % (epoch + 1, mean_true_loss, std_true_loss, mean_false_loss, std_false_loss)))

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

                            test_acc_list.append(test_accuracy)
                            f1_macro_list.append(f1_macro)
                            f1_weighted_list.append(f1_weighted)
                            f1_micro_list.append(f1_micro)

                        mean_test_acc = round(np.mean(test_acc_list), 4)
                        mean_f1_macro = round(np.mean(f1_macro_list), 4)
                        mean_f1_weighted = round(np.mean(f1_weighted_list), 4)
                        mean_f1_micro = round(np.mean(f1_micro_list), 4)
                        with open(total_file, "a") as myfile:
                            if use_tsda:
                                statement = '%s_%s%.2f_startrefurb%d_len%d' % (args.dataset, args.noise_type, args.label_noise_rate, args.start_refurb, args.refurb_len)
                            else:
                                statement = '%s_%s%.2f_startrefurb%d_notsda' % (args.dataset, args.noise_type, args.label_noise_rate, args.start_refurb)
                            myfile.write('%s,%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f\n' % (statement, str(test_acc_list), mean_test_acc,
                                          str(f1_macro_list), mean_f1_macro, str(f1_weighted_list),mean_f1_weighted, str(f1_micro_list), mean_f1_micro))

if __name__ == '__main__':
    # main(archive='UEA', gpu_id=0, noise_type='symmetric', noise_rates=[0.2], start_refurb=30)
    # main(archive='other', gpu_id=0, noise_type='symmetric', noise_rates=[0.2], start_refurb=30)
    # main(archive='other', gpu_id=0, noise_type='symmetric', noise_rates=[0.2], start_refurb=30, use_tsda=False)
    # main(archive='other', gpu_id=0, noise_type='symmetric', noise_rates=[0.2], start_refurb=100)

    # main(archive='UEA', gpu_id=1, noise_type='symmetric', noise_rates=[0.5], start_refurb=30)
    # main(archive='other', gpu_id=1, noise_type='symmetric', noise_rates=[0.5], start_refurb=30)
    # main(archive='other', gpu_id=1, noise_type='symmetric', noise_rates=[0.5], start_refurb=30, use_tsda=False)
    # main(archive='other', gpu_id=1, noise_type='symmetric', noise_rates=[0.5], start_refurb=100)
    #
    # main(archive='UEA', gpu_id=2, noise_type='instance', noise_rates=[0.4], start_refurb=30)
    # main(archive='other', gpu_id=2, noise_type='instance', noise_rates=[0.4], start_refurb=30)
    # main(archive='other', gpu_id=2, noise_type='instance', noise_rates=[0.4], start_refurb=30, use_tsda=False)
    # main(archive='other', gpu_id=2, noise_type='instance', noise_rates=[0.4], start_refurb=100)

    # main(archive='UEA', gpu_id=3, noise_type='pairflip', noise_rates=[0.4], start_refurb=30)
    # main(archive='other', gpu_id=3, noise_type='pairflip', noise_rates=[0.4], start_refurb=30)
    # main(archive='other', gpu_id=3, noise_type='pairflip', noise_rates=[0.4], start_refurb=30, use_tsda=False)
    # main(archive='other', gpu_id=3, noise_type='pairflip', noise_rates=[0.4], start_refurb=100)

    # main(archive='other', gpu_id=3, noise_type='symmetric', noise_rates=[0.2], start_refurb=30, lams=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # main(archive='other', gpu_id=0, noise_type='symmetric', noise_rates=[0.5], start_refurb=30, lams=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # main(archive='other', gpu_id=1, noise_type='instance', noise_rates=[0.4], start_refurb=30, lams=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # main(archive='other', gpu_id=2, noise_type='pairflip', noise_rates=[0.4], start_refurb=30, lams=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    main(archive='other', gpu_id=0, noise_type='instance', noise_rates=[0.4])
