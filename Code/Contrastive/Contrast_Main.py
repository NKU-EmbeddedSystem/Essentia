import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append('.')
from EEGDiff.Code.utils import train_finetune_test_split
from EEGDiff.Code.Dataset import EEGContrastDataSet, EEGClassifierDataSet, dataload
from EEGDiff.Code.Contrastive.Contrastive_Model import Encoder, Classifier
from EEGDiff.Code.Contrastive.Contrastive_utils import InfoNCELoss, generate_negative_index, freq_disturb



def train(model, dataloader, optimizer, device, pure_eeg, noise_eeg, temperature, neg_size):
    model.to(device)
    model.train()
    contrast_loss = InfoNCELoss(temperature).to(device)
    
    losses = []

    for (eeg_arg, eeg_noise, index) in tqdm(dataloader):
        optimizer.zero_grad()

        # 生成负样本集合
        eeg_neg_pure = pure_eeg[generate_negative_index(index, neg_size)]
        eeg_neg_noise = noise_eeg[generate_negative_index(index, neg_size)]
        eeg_neg_pure = freq_disturb(eeg_neg_pure)
        eeg_neg_noise = freq_disturb(eeg_neg_noise)
        eeg_neg = torch.cat((eeg_neg_pure, eeg_neg_noise), dim=0)

        eeg_arg = eeg_arg.to(device).float()
        eeg_noise = eeg_noise.to(device).float()
        eeg_neg = eeg_neg.to(device).float()

        eeg_arg_embed = model(eeg_arg)
        eeg_noise_embed = model(eeg_noise)
        eeg_neg_embed = model(eeg_neg)

        loss = contrast_loss(eeg_arg_embed, eeg_noise_embed, eeg_neg_embed)
        loss_rev = contrast_loss(eeg_noise_embed, eeg_arg_embed, eeg_neg_embed)
        loss = 0.5 * (loss + loss_rev)
        
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())

    losses = np.array(losses)
    return losses.mean().item()


def test(model, dataloader, device, pure_eeg, noise_eeg, temperature, neg_size):
    model.to(device)
    model.eval()
    contrast_loss = InfoNCELoss(temperature).to(device)
    
    losses = []

    for (eeg_arg, eeg_noise, index) in tqdm(dataloader):
        # 生成负样本集合
        eeg_neg_pure = pure_eeg[generate_negative_index(index, neg_size, 2999)]
        eeg_neg_noise = noise_eeg[generate_negative_index(index, neg_size, 2999)]
        eeg_neg_pure = freq_disturb(eeg_neg_pure)
        eeg_neg_noise = freq_disturb(eeg_neg_noise)
        # print(eeg_neg_pure.shape, eeg_neg_noise.shape)
        eeg_neg = torch.cat((eeg_neg_pure, eeg_neg_noise), dim=0)

        eeg_arg = eeg_arg.to(device).float()
        eeg_noise = eeg_noise.to(device).float()
        eeg_neg = eeg_neg.to(device).float()

        eeg_arg_embed = model(eeg_arg)
        eeg_noise_embed = model(eeg_noise)
        eeg_neg_embed = model(eeg_neg)

        loss = contrast_loss(eeg_arg_embed, eeg_noise_embed, eeg_neg_embed)
        loss_rev = contrast_loss(eeg_noise_embed, eeg_arg_embed, eeg_neg_embed)
        loss = 0.5 * (loss + loss_rev)
        losses.append(loss.cpu().detach().numpy())

    losses = np.array(losses)
    return losses.mean().item()


def evaluate(model, finetune_dataloader, test_dataloader, optimizer, device):
    model.to(device)
    critation = nn.CrossEntropyLoss()

    # model finetuning for motivation recognition
    model.train()
    finetuning_losses = 0.0
    # for i in range(5):
    for (data, label) in tqdm(finetune_dataloader):
        data = data.to(device).float()
        label = label.to(device).float()
        optimizer.zero_grad()
        pre = model(data)
        finetuning_loss = critation(pre, label)
        finetuning_loss.backward()
        optimizer.step()
        finetuning_losses += finetuning_loss.item()
    finetuning_losses /= len(finetune_dataloader)

    # compute accuracy after finetuning
    model.eval()
    accs = 0.0
    with torch.no_grad():
        for (data, label) in tqdm(test_dataloader):
            data = data.to(device).float()
            label = label.to(device).float()

            label = torch.argmax(label, dim=-1)
            pred = model(data)
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=-1)

            correct = torch.eq(pred, label)
            acc = correct.sum().float().item() / len(label)
            accs += acc
        accs /= len(test_dataloader)

    return finetuning_losses, accs


def main():
    BATCH_SIZE = 1024
    NEGATIVE_SIZE = 1024
    EPOCHS = 500
    TEMPERATURE = 0.5
    # max_acc = 0.5
    mini_loss = 50


    print("torch.cuda.is_availabel(): ", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # 读入数据
    # [train, finetune, test]
    noise_eeg, pure_eeg, label = dataload('/home/zzb/EEGDiff/Data/1c_win_shuffle')

    # 对比学习数据集构建
    train_dataset = EEGContrastDataSet(pure_eeg[0], noise_eeg[0])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 下游任务数据集构建
    # finetune_dataset = EEGClassifierDataSet(noise_eeg[1], label[1])
    # finetune_loader = DataLoader(finetune_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset =  EEGContrastDataSet(pure_eeg[2], noise_eeg[2])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model initialization
    encoder = Encoder()
    classifier_model = Classifier(encoder)
    
    # optimizer initialization
    train_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    finetuning_optimizer = torch.optim.Adam(classifier_model.parameters(), lr=1e-4)

    # main loop 
    for epoch in range(EPOCHS):
        train_loss = train(encoder, train_loader, train_optimizer, device, pure_eeg[0], noise_eeg[0], TEMPERATURE, NEGATIVE_SIZE)
        test_loss = test(encoder, test_loader, device, pure_eeg[2], noise_eeg[2], TEMPERATURE, NEGATIVE_SIZE)

        '''
        # finetuning_loss, test_acc = evaluate(classifier_model, finetune_loader, test_loader, finetuning_optimizer,  device)

        # tqdm.write(f'epoch {epoch + 1}, train_loss: {train_loss:.8f}, finetuning_loss: {finetuning_loss:.8f}, test_acc: {test_acc:.3f}')
        
        # with open('/home/zzb/EEGDiff/Log/eeg_contrastive_log.txt', 'a') as f:
        #     f.write(f'epoch {epoch + 1}, train_loss: {train_loss:.8f}, finetuning_loss: {finetuning_loss:.8f}, test_acc: {test_acc:.3f}')

        # if max_acc <= test_acc:
        #     max_acc = test_acc
        #     torch.save(encoder.state_dict(), f'/home/zzb/EEGDiff/Model/contrastive/encoder_{epoch+1}_{test_acc:.3f}.pkl')
        '''
  

        tqdm.write(f'epoch {epoch + 1}, train_loss: {train_loss:.8f}, test_loss: {test_loss:.8f}')
        
        with open('/home/zzb/EEGDiff/Log/eeg_contrastive_log.txt', 'a') as f:
            f.write(f'epoch {epoch + 1}, train_loss: {train_loss:.8f}, test_loss: {test_loss:.8f}\n')

        if test_loss <= mini_loss:
            mini_loss = test_loss
            torch.save(encoder.state_dict(), f'/home/zzb/EEGDiff/Model/contrastive/encoder_{epoch+1}_{test_loss:.8f}.pkl')



        


if __name__ == '__main__':
    main()
    
