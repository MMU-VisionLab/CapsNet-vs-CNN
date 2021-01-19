'''
Helper functions.
'''
import os
import shutil
from tqdm import tqdm
import torch
from plot import plot_reconstruction

def check_dir_exists(dir_path):
    '''
    Checks if a given directory exists.
    '''
    return os.path.exists(dir_path)

def del_dir(dir_path):
    '''
    Deletes the given directory.
    '''
    shutil.rmtree(dir_path)
    return None

def create_dir(dir_path):
    '''
    Creates a directory at the given path.
    '''
    os.makedirs(dir_path)
    return None

def onehot_encode(tensor, num_classes, device):
    '''
    Encodes the given tensor into one-hot vectors.
    '''
    return torch.eye(num_classes).to(device).index_select(dim=0, index=tensor.to(device))


def run_cnn_model(generator, model, criterion, optimizer, lr_decayer, device, train=True):
    '''
    Use this func either to run one epoch of training or testing for the cnn model with the given data.
    '''

    epoch_loss = 0
    epoch_accuracy = 0
    i = 0
    for i, sample in tqdm(enumerate(generator)):

        batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)

        net_output = model(batch_x)

        if train:
            loss = model.optimize_model(predicted=net_output, target=batch_y, loss_func=criterion, optim_func=optimizer, decay_func=lr_decayer)
        else:
            loss = model.calculate_loss(predicted=net_output, target=batch_y, loss_func=criterion)

        epoch_loss += loss
        epoch_accuracy += model.calculate_accuracy(net_output, batch_y)


    return epoch_loss, epoch_accuracy, i


def run_capsnet_model(generator, model, criterion, optimizer, lr_decayer, device, train=True):
    '''
    Use this func either to run one epoch of training or testing for the cnn model with the given data.
    '''

    epoch_loss = 0
    epoch_accuracy = 0
    i = 0
    for i, sample in tqdm(enumerate(generator)):

        batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)

        v_lengths, decoded_images = model(batch_x, batch_y)

        if train:
            loss = model.optimize_model(predicted=v_lengths, target=batch_y, ori_imgs=batch_x, decoded=decoded_images, loss_func=criterion,
                                            optim_func=optimizer, decay_func=lr_decayer, lr_decay_step=False)
        else:
            loss = model.calculate_loss(predicted=v_lengths, target=batch_y, ori_imgs=batch_x, decoded=decoded_images,
                                              loss_func=criterion)

        epoch_loss += loss
        epoch_accuracy += model.calculate_accuracy(v_lengths, batch_y)

    return epoch_loss, epoch_accuracy, i


def run_deepcaps_model(epoch_idx, generator, model, criterion, optimizer, lr_decayer, num_classes, device, train=True):
    '''
    Use this func either to run one epoch of training or testing for the deep capsnet model with the given data.
    '''

    epoch_loss = 0
    epoch_accuracy = 0
    i = 0
    batch_x, batch_y, reconstructed, indices = None, None, None, None
    for i, sample in tqdm(enumerate(generator)):

        batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)

        onehot_label = onehot_encode(batch_y, num_classes=num_classes, device=device)

        outputs, _, reconstructed, indices = model(batch_x, onehot_label)

        if train:
            loss = model.optimize_model(predicted=outputs, target=onehot_label, ori_imgs=batch_x, decoded=reconstructed,
                                        loss_func=criterion, optim_func=optimizer, decay_func=lr_decayer, lr_decay_step=False)
        else:
            loss = model.calculate_loss(predicted=outputs, target=onehot_label, ori_imgs=batch_x, decoded=reconstructed, loss_func=criterion)

        epoch_loss += loss
        epoch_accuracy += model.calculate_accuracy(predictions=indices, labels=batch_y)

    plot_reconstruction(path='./graphs_folder/', num_epoch=epoch_idx, original_images=batch_x.detach(), reconstructed_images=reconstructed.detach(),
                        predicted_classes=indices.detach(), true_classes=batch_y.detach())

    return epoch_loss, epoch_accuracy, i

