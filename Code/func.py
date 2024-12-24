import torch
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn

import os
import sys

import Contrastive.Contrastive_Model as Contrastive
from Test.MMCNN.MMCNNModel import MMCNNModel1
import Diffusion.Diffusion_Model as Diffusion_Model
import Diffusion.Diffusion_Diffusion as Diffusion_Diffusion
import Contrastive.Contrastive_utils as Contrastive_utils


def evaluate(model, data, label, device):
    model.to(device)
    model.eval()
    data = data.to(device).float()
    label = label.to(device).float()
    label = torch.argmax(label, dim=-1)
    pred = model(data)
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=-1)
    correct = torch.eq(pred, label)
    acc = correct.sum().float().item() / len(label)
    return acc


def diffusion_data_load(root_path, batch_size):
    """
    this function help you to load the data in following step:
    1. load pure eeg data, noise eeg data, label for train and test
    2. build the torch dataset and dataloader
    """

    train_noise_eeg = None
    train_pure_eeg = None
    train_label_eeg = None
    test_noise_eeg = None
    test_pure_eeg = None
    test_label_eeg = None

    train_torch_dataset = Data.TensorDataset(train_noise_eeg, train_pure_eeg)
    train_loader = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_torch_dataset = Data.TensorDataset(test_noise_eeg, test_pure_eeg, test_label_eeg)
    test_loader = Data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def model_setting_contrastive(guided_sr3, guided_ldm, test_model_path, device):
    mmcnn_ckp = torch.load(test_model_path)
    test_model = MMCNNModel1().to(device)
    test_model.load_state_dict(mmcnn_ckp)

    contrastive_model = Contrastive.ContrastiveSemanticTF_New().to(device)  # latest！！！
    contrastive_loss_model = Contrastive_utils.InfoNCELoss().to(device)

    model = Diffusion_Model.Unet_NKDM_SEM(      # latest!!
        dim=32,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        guided_sr3=guided_sr3,
        latent_diffusion=guided_ldm,
    )

    diffusion = Diffusion_Diffusion.GaussianDiffusion_E2ELoss(
        model=model,
        image_size=(1, 1000),
        timesteps=1000,
        sampling_timesteps=10,
        ddim_sampling_eta=0.5,
    ).to(device)

    return diffusion, test_model, contrastive_model, contrastive_loss_model


def loss_fn(noise, pre_nosie, x0, pre_x0, e2e_scale=0.5, ):
    loss1 = F.mse_loss(noise, pre_nosie)
    loss2 = F.mse_loss(x0, pre_x0)
    loss = loss1 + e2e_scale * loss2

    return loss


def contrastive_loss_fn(noise, pre_nosie, query, positive, negative, contrastive_loss_model, contrastive_scale=0.1):
    nkdm_loss = F.mse_loss(noise, pre_nosie)
    contrastive_loss = contrastive_loss_model(query, positive, negative)
    loss = nkdm_loss + contrastive_scale * contrastive_loss

    return loss



