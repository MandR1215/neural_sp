import torch
from torch.optim import Adam
from neuralsd import NeuralSD
from preference_generator import Euclidean
from utils import preference_to_matrix
import numpy as np
from layers import RowCrossEntropyLoss
from algorithm import MatchingAlgorithm, DeferredAcceptance, EqualWeightedHungarian, MinorityHungarian
from train import train
from data import MatchingDataset, create_loader
import random
from argparse import ArgumentParser
import datetime
import json
from tqdm import tqdm

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    g = torch.Generator()
    g.manual_seed(seed)

    return g

def create_example_batch(n_data:int=100, input_size:int=10, n_workers:int=100, n_firms:int=100, matcher_class:MatchingAlgorithm=DeferredAcceptance):
    batch = []
    for _ in tqdm(range(n_data)):
        feature_workers = torch.normal(mean=-1.0, std=1.0, size=(n_workers, input_size))
        feature_firms = torch.normal(mean=1.0, std=1.0, size=(n_firms, input_size))
        X = torch.cat([feature_workers, feature_firms])
        worker_preferences = Euclidean(n_agents=n_workers, 
                                       n_counterparts=n_firms, 
                                       attributes=(feature_workers, feature_firms),
                                       unmatch_threshold=8).get_all()
        firm_preferences = Euclidean(n_agents=n_firms, 
                                     n_counterparts=n_workers, 
                                     attributes=(feature_firms, feature_workers),
                                     unmatch_threshold=8).get_all()

        y = matcher_class(n_workers=n_workers, n_firms=n_firms).match(worker_preferences=worker_preferences, firm_preferences=firm_preferences)
        y = torch.tensor(y).float()
        batch.append(((worker_preferences,firm_preferences,X),y))
    return batch

def main(args):
    # matching label
    matching_models = {
        'DA': DeferredAcceptance,
        'EH': EqualWeightedHungarian,
        'MH': MinorityHungarian
    }

    # set seed
    g = set_seed(args.seed)

    # train data
    print("prepare train data")
    input_size    = args.input_size
    n_workers     = args.n_workers
    n_firms       = args.n_firms
    train_size    = args.train_size
    device        = args.device
    batch_size    = args.batch_size
    matcher_class = matching_models[args.matcher]
    train_batch = create_example_batch(n_data=train_size,
                                       input_size=input_size, 
                                       n_workers=n_workers,
                                       n_firms=n_firms,
                                       matcher_class=matcher_class)
    train_dataset = MatchingDataset(data=train_batch)
    train_dataloader = create_loader(dataset=train_dataset, 
                                     batch_size=batch_size,
                                     worker_init_fn=seed_worker,
                                     generator=g)
    
    # model
    print("prepare model")
    pow = args.pow
    tau = args.tau
    model_name = datetime.datetime.today().strftime('%Y_%m_%d--%H_%M_%S')
    model = NeuralSD(input_size=input_size, pow=pow, tau=tau).to(device=device)

    # train
    print("start train")
    epochs = args.epochs
    lam = args.lam
    optimizer = Adam(params=model.parameters(), lr=0.01)
    criterion = RowCrossEntropyLoss()
    train(model=model, 
          criterion=criterion, 
          optimizer=optimizer, 
          data_loader=train_dataloader, 
          epochs=epochs,
          lam=lam)
    
    torch.save(model.state_dict(), f'models/model_{model_name}.pt')

    with open(f"configs/args_{model_name}.json", 'w') as f:
        json.dump(vars(args), f)

   
if __name__=='__main__':
    args = ArgumentParser()
    args.add_argument('--seed',
                      type=int,
                      default=42,
                      help='random seed.')
    args.add_argument('--n_workers', 
                      type=int, 
                      default=10,
                      help='Number of workers. (default: 10)')
    args.add_argument('--n_firms', 
                      type=int, 
                      default=10,
                      help='Number of firms. (default: 10)')
    args.add_argument('--batch_size',
                      type=int,
                      default=4,
                      help='batch size')
    args.add_argument('--train_size',
                      type=int,
                      default=1000,
                      help='# of train data.')
    args.add_argument('--input_size',
                      type=int,
                      default=10,
                      help='dimension of feature vector.')
    args.add_argument('--tau',
                      type=float,
                      default=0.1,
                      help='temperature parameter for SoftSort.')
    args.add_argument('--lam',
                      type=float,
                      default=0.0,
                      help='lambda parameter to stability regularization.')
    args.add_argument('--pow',
                      type=float,
                      default=1.0,
                      help='p parameter for SoftSort.')
    args.add_argument('--epochs',
                      type=int,
                      default=30,
                      help='# of training epochs')
    args.add_argument('--device',
                      choices=['cpu', 'cuda'],
                      default='cpu',
                      help='Device of model and data.')
    args.add_argument('--matcher',
                      choices=['DA', 'EH', 'MH'],
                      default='DA',
                      help='Matching model. (default: DA)')

    main(args.parse_args())
