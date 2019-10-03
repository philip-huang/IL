""" Generative VCL"""
import numpy as np
import torch
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import trainer
import mnist
import utils
from model.bayesian_generator import *

def fit(args, model, device, optimizer, loss_fn, dataset, labels_list, task_id):
    # Dataloader
    train_loader = trainer.get_loader(mnist.getTrain(dataset), args, device, 'train')
    val_loader = trainer.get_loader(mnist.getVal(dataset), args, device, 'val')
    # Log Best Accuracy
    best_val_loss = 0
    # Early Stopping
    early_stop = 0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Prepare model for current task
        model.set_task_id(labels_list[task_id])
        trainer.train(args, model, device, train_loader, optimizer, epoch, loss_fn)
        val_loss, _ = trainer.test(args, model, device, val_loader, loss_fn, val=True)
        if val_loss > best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= args.early_stop_after:
                break
    
    return best_state

def get_dataset(args, task_id, split):
    trans = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
    if args.dataset=='splitMNIST10':
        Dataset = mnist.SplitMNIST10

    if split == 'train':
        trainval = Dataset('data', task_id, train=True, download=True, transform=trans)
        dataset = mnist.getTrain(trainval)
    elif split == 'val':
        trainval = Dataset('data', task_id, train=True, download=True, transform=trans)
        dataset = mnist.getVal(trainval)
    elif split == 'trainval':
        dataset = Dataset('data', task_id, train=True, download=True, transform=trans)
    else:
        dataset = Dataset('data', task_id, train=False, download=True, transform=trans)
    return dataset

def test_all_tasks(args, model, device, loss_fn, labels_list, task_id, split):
    # Evaluation Loop
    ims = []
    for j, _ in enumerate(labels_list[:task_id]):
        # Evaluate on the test set
        testset = get_dataset(args, j, split)
        test_loader = trainer.get_loader(testset, args, device, split)
        model.set_task_id(j)
        # Generate Image
        with torch.no_grad():
            latent, image = model.generator.generate(num=100)
        ims.append(image)
    return ims


def run_vcl(args, device, labels_list):
    # TODO: Add classifier accuracy
    all_vis_images = []

    # train all tasks incrementally
    model, _ = trainer.get_model(args, device)
    for task_id, labels in enumerate(labels_list):
        print ("===========TASK {}: {}=============".format(task_id + 1, labels))
        # Model
        old_model = model
        model, model_name = trainer.get_model(args, device, task_id=task_id)
        model.load_state_dict(old_model.state_dict(), strict=False)
        
        # Loss Function and Optimizer
        loss_fn = trainer.get_loss_fn(args, device, model, old_model)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Dataset
        dataset = get_dataset(args, task_id, 'trainval')

        # Fit
        best_state = fit(args, model, device, optimizer, loss_fn, dataset, labels_list, task_id)
        
        # Generate some images
        ims = test_all_tasks(args, model, device, loss_fn, labels_list, task_id, split=True)
        for test_task_id, im in enumerate(ims):
            utils.save_generate_ims(im, 'mnist_gen', task_id, test_task_id)
        all_vis_images.extend(select_and_pad_ims(ims))
        
        # save model
        path = trainer.get_model_path(labels, model_name)
        torch.save(best_state, path)
    
    # Save Final Image
    final_vis_im = torch.cat(all_vis_images)
    utils.save_generated_ims(final_vis_im, 'mnist_gen', 'final', 'all')
    
    return

def select_and_pad_ims(ims):
    empty_im = torch.zeros(1, 28, 28)
    selected = [empty_im for _ in range(10)]

    for task_id, im in enumerate(ims):
        batch_size = ims.size(0)
        im = ims[np.random.randint(batch_size)]
        selected[task_id] = im
    
    return selected