""" VCL Coreset Module"""
import numpy as np
import torch
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms

import trainer
import mnist

def fit(args, model, device, optimizer, loss_fn,  coresets, dataset, labels_list, task_id):
    # Dataloader
    train_loader = trainer.get_loader(mnist.getTrain(dataset), args, device, 'train')
    val_loader = trainer.get_loader(mnist.getVal(dataset), args, device, 'val')
    # Log Best Accuracy
    best_val_acc = 0
    all_test_accs = []
    # Early Stopping
    early_stop = 0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Prepare model for current task
        model.set_range(labels_list[task_id])
        trainer.train(args, model, device, train_loader, optimizer, epoch, loss_fn)
        _, val_acc = trainer.test(args, model, device, val_loader, loss_fn, val=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= args.early_stop_after:
                break

        # Evaluate all tasks on test set
        test_accs = test_all_tasks(coresets, args, model, device, 
            loss_fn, labels_list, 'test')
        all_test_accs.append(test_accs)
    
    return best_state, all_test_accs

def get_dataset(args, task_id, split):
    trans = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
    if args.dataset=='splitMNIST':
        Dataset = mnist.SplitMNIST
    elif args.dataset=='fashionMNIST':
        Dataset = mnist.FashionMNIST
    elif args.dataset=='permutedMNIST':
        Dataset = mnist.PermutedMNIST
    
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

def test_all_tasks(coresets, args, model, device, loss_fn, labels_list, split):
    def fit_coreset(coreset, labels, args, model, device):
        # Get Inference Model
        final_model, _ = trainer.get_model(args, device)
        final_model.load_state_dict(model.state_dict())
        final_model.set_range(labels)
        
        # Get Loss Function and Optimizer and Dataloader
        loss_fn = trainer.get_loss_fn(args, device, final_model, model)
        optimizer = optim.Adam(final_model.parameters(), lr=args.lr)
        coreset_loader = trainer.get_loader(coreset, args, device, 'coreset')
        
        for epoch in range(args.coreset_epochs):
            trainer.train(args, final_model, device, coreset_loader, optimizer, epoch, loss_fn, verbose=False)

        return final_model

    # Coreset Evaluation Loop
    test_accs = np.zeros(len(labels_list))
    for j, labels in enumerate(labels_list):
        if args.coreset_size > 0 and len(coresets) > j:
            # Get Inference Model by Tuning on Coreset
            final_model = fit_coreset(coresets[j], labels, args, model, device)
        else:
            # No Coreset
            final_model = model

        # Evaluate on the test set
        testset = get_dataset(args, j, split)
        test_loader = trainer.get_loader(testset, args, device, split)
        final_model.set_range(labels)
        _, test_accs[j] = trainer.test(args, final_model, device, test_loader, loss_fn)

    return test_accs

def run_vcl(args, device, labels_list):
    task_final_accs = np.zeros((5, 5)) # Test accuracy after each task ends
    all_accs = [] # Test accuracy after every epoch
    
    coresets = [] # Coresets
    coreset_method = Coreset.rand_from_batch

    # Pretraining
    print ("=============Pretraining ==================")
    model, name = trainer.get_model(args, device, mle=True)
    loss_fn = trainer.get_loss_fn(args, device, model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dataset = get_dataset(args, 0, 'trainval')
    fit(args, model, device, optimizer, loss_fn, coresets, dataset, labels_list, 0)

    # train all tasks incrementally
    for task_id, labels in enumerate(labels_list):
        print ("===========TASK {}: {}=============".format(task_id + 1, labels))
        # Model
        old_model = model
        model, model_name = trainer.get_model(args, device)
        model.load_state_dict(old_model.state_dict())
        # Loss Function and Optimizer
        loss_fn = trainer.get_loss_fn(args, device, model, old_model)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Dataset and Coreset
        dataset = get_dataset(args, task_id, 'trainval')
        if coreset_method is not None:
            coresets, dataset = coreset_method(coresets, dataset, args.coreset_size)

        # Fit
        best_state, test_accs = fit(args, model, device, optimizer, loss_fn, coresets, dataset, labels_list, task_id)
        task_final_accs[task_id] = test_accs[-1]
        all_accs.extend(test_accs)

        # save model
        path = trainer.get_model_path(labels, model_name)
        torch.save(best_state, path)

    return task_final_accs, all_accs

class Coreset():
    """ Random coreset selection """
    @staticmethod
    def rand_from_batch(coresets, dataset, coreset_size):
        sizes = [coreset_size, len(dataset) - coreset_size]
        new_coreset, dataset = data.random_split(dataset, sizes)
        coresets.append(new_coreset)

        return coresets, dataset

    """ K-center coreset selection """
    @staticmethod
    def k_center(coresets, dataset, coreset_size):
        # Select K centers from dataset and add to current coreset
        dists = np.full(len(dataset), np.inf)
        current_id = 0
        dists = Coreset.update_distance(dists, dataset, current_id)
        idx = [ current_id ]
        
        for i in range(1, coreset_size):
            current_id = np.argmax(dists)
            dists = Coreset.update_distance(dists, dataset, current_id)
            idx.append(current_id)
        
        keep_idx = [i for i in range(len(dataset)) if i not in idx]
        new_coreset = data.Subset(dataset, idx)
        coresets.append(new_coreset)
        dataset = data.Subset(dataset, keep_idx)
        
        return coresets, dataset

    @staticmethod
    def update_distance(dists, dataset, current_id):
        current_datapoint, _ = dataset[current_id]
        for i in range(len(dataset)):
            data, _ = dataset[i]
            current_dist = np.linalg.norm(data - current_datapoint)
            dists[i] = np.minimum(current_dist, dists[i])
        return dists