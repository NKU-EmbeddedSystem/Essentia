import argparse
import sys
import os
working_dir = os.getcwd()
sys.path.append(working_dir)
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from func import evaluate, diffusion_data_load, model_setting_semanticskipca, loss_fn, \
    model_setting_contrastive, contrastive_loss_fn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='', type=str, help='dataset to use')
args = parser.parse_args() 

# global var
DATASET_NAME = args.dataset
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
MODEL_NAME = ''
EPOCHS = 1000
GUIDED_SR3 = False
GUIDED_LDM = True
LOSS_SCALE = 1 / 2

if DATASET_NAME == 'your_dataset':
    DATA_ROOT_PATH = None   # path to your dataset
    TEST_MODEL_PATH = None  # path to test model for your dataset for evaluation
else:
    raise ValueError('dataset not found')

def main(): 
    max_acc = 0.5
    train_loader, test_loader = diffusion_data_load(root_path=DATA_ROOT_PATH, batch_size=BATCH_SIZE)
    diffusion, test_model, contrastive_model, info_loss_model = model_setting_contrastive(GUIDED_SR3, GUIDED_LDM, TEST_MODEL_PATH, DEVICE)
    optimizer = torch.optim.Adam(list(diffusion.parameters()) + list(contrastive_model.parameters()), lr=1e-5, betas=(0.9, 0.99))

    negative_set = []
    for (train_noise_eeg, _) in train_loader:
        negative_set.append(train_noise_eeg)

    for epoch in range(0, EPOCHS):
        train_loss_list = []
        test_loss_list = []
        test_acc_list = []

        for i, (train_noise_eeg, train_pure_eeg) in enumerate(tqdm(train_loader, ncols=100)):
            diffusion.train()
            contrastive_model.train()
            train_noise_eeg = train_noise_eeg.float().to(DEVICE)
            train_pure_eeg = train_pure_eeg.float().to(DEVICE)
            optimizer.zero_grad()

            '''
            # contrastive compute
            # noise_eeg <-> pure_eeg => positive
            # noise_eeg <-> random_eeg => negative
            '''

            # random_eeg = torch.rand_like(train_noise_eeg).to(DEVICE)
            neg_eeg = torch.cat(negative_set[0:i] + negative_set[i + 1:], dim=0)
            random_indices = torch.randperm(neg_eeg.size(0))[:512]
            neg_eeg = neg_eeg[random_indices].float().to(DEVICE)

            emb_train_noise_eeg = contrastive_model(train_noise_eeg)
            emb_train_pure_eeg = contrastive_model(train_pure_eeg)
            emb_train_random_eeg = contrastive_model(neg_eeg)

            noise, pre_noise = diffusion.contrastive_forward(train_noise_eeg, train_pure_eeg, emb_train_noise_eeg)
            train_loss = contrastive_loss_fn(noise=noise, pre_nosie=pre_noise, query=emb_train_noise_eeg,
                                             positive=emb_train_pure_eeg, negative=emb_train_random_eeg,
                                             contrastive_loss_model=info_loss_model)

            train_loss.backward()
            optimizer.step()
            diffusion.zero_grad()
            train_loss_list.append(train_loss.cpu().detach().numpy())

        if (epoch >= 450) or (epoch % 100 == 99):
            for (test_noise_eeg, test_pure_eeg, test_label) in tqdm(test_loader, ncols=100):
                diffusion.eval()
                contrastive_model.eval()
                test_noise_eeg = test_noise_eeg.float().to(DEVICE)
                test_pure_eeg = test_pure_eeg.float().to(DEVICE)
                test_label = test_label.float().to(DEVICE)

                semantic = contrastive_model(test_noise_eeg)
                test_sampel = diffusion.contrastive_sample(test_noise_eeg, semantic)
                test_loss = F.mse_loss(test_sampel, test_pure_eeg)
                test_loss_list.append(test_loss.cpu().detach().numpy())
                acc = evaluate(test_model, test_sampel, test_label, device=DEVICE)
                test_acc_list.append(acc)

        test_acc_list = np.array(test_acc_list)

        tqdm.write(
            f'epoch:{epoch + 1} -> train_loss:{np.mean(train_loss_list):.8f}, test_loss:{np.mean(test_loss_list):.8f}, test_acc:{np.mean(test_acc_list):.8f}')
        


        with open(working_dir + MODEL_NAME + '_' + DATASET_NAME + '_log.txt', 'a') as f:
            f.write(
                f'epoch:{epoch + 1} -> train_loss:{np.mean(train_loss_list):.8f} , test_loss:{np.mean(test_loss_list):.8f}, test_acc:{np.mean(test_acc_list):.8f}\n')

        # save base classifier acc of sampling
        optimizer_save = optimizer.state_dict()
        save_dict = {
            'NKDM': diffusion.state_dict(),
            'contrastive': contrastive_model.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer_save
        }

        if not os.path.exists(working_dir + DATASET_NAME + '/' + MODEL_NAME):
            os.makedirs(working_dir + DATASET_NAME + '/' + MODEL_NAME)

        if np.mean(test_acc_list) > max_acc:
            print('save max_acc model')
            best_epoch = epoch + 1
            max_acc = np.mean(test_acc_list)
            torch.save(save_dict, working_dir + DATASET_NAME + '/' + MODEL_NAME + f'/zdm_{epoch + 1}_{np.mean(test_acc_list):.5f}.pkl')


if __name__ == '__main__':
    main()
