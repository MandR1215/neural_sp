import torch
from neuralsd import NeuralSD
from preference_generator import Euclidean
from utils import preference_to_matrix, matrix_to_preference, hamming_distance, number_of_blocking_pairs, preference_to_rational
import numpy as np
from algorithm import MatchingAlgorithm, DeferredAcceptance, RankHungarian, EqualWeightedHungarian, MinorityHungarian, SerialDictatorship
from scipy import stats
import random
from argparse import ArgumentParser
import pprint
from itertools import permutations
from collections import defaultdict, Counter
import pandas as pd
from math import factorial
from loss_functions import compute_st_unit
from tqdm import tqdm
import datetime
import pickle

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
    matchers = []
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
        
        matcher = matcher_class(n_workers=n_workers, n_firms=n_firms)
        y = matcher.match(worker_preferences=worker_preferences, firm_preferences=firm_preferences)
        y = torch.tensor(y).float()
        batch.append(((worker_preferences,firm_preferences,X),y))
        
        matchers.append(matcher)
    return batch, matchers

def compute_optimal_sd(test_batch):
    optimal_hamming_distances = []
    optimal_rankings = []

    for data in test_batch:
        worker_preferences = data[0][0]
        firm_preferences = data[0][1]

        n, m = len(worker_preferences), len(firm_preferences)
        all_rankings = list(permutations(list(range(n+m))))
        all_hamms = []
        
        for ranking in all_rankings:
            sd = SerialDictatorship(n_workers=n, n_firms=m, ranking=ranking)
            matching = sd.match(worker_preferences=worker_preferences, firm_preferences=firm_preferences)
            all_hamms.append(hamming_distance(data[1].numpy(), matching))
        min_hamm = min(all_hamms)
        argmin_hamm = np.where(all_hamms == np.min(all_hamms))[0]

        optimal_hamming_distances.append(min_hamm)
        optimal_rankings.append(set(all_rankings[i] for i in argmin_hamm))

    return optimal_hamming_distances, optimal_rankings

def inclusion_rate(test_batch, optimal_rankings, predicted_rankings) -> pd.DataFrame:
    inclusion = defaultdict(lambda: list())

    for i, data in enumerate(test_batch):
        ground_truth = optimal_rankings[i]
        pred = tuple(torch.argmax(predicted_rankings[i], dim=0).tolist())
        inclusion[len(ground_truth)].append(pred in ground_truth)

    inclusion = dict([(key, sum(inclusion[key])) for key in inclusion.keys()])
    inclusion = dict((key, inclusion[key]) for key in sorted(inclusion.keys()))
    return inclusion

def main(args):
    # matching label
    matching_models = {
        'DA': DeferredAcceptance,
        'EH': EqualWeightedHungarian,
        'MH': MinorityHungarian
    }
    # set seed
    g = set_seed(args.seed)

    # test data
    print("prepare test data")
    input_size    = args.input_size
    n_workers     = args.n_workers
    n_firms       = args.n_firms
    test_size     = args.test_size
    matcher_class = matching_models[args.matcher]
    test_batch, _ = create_example_batch(n_data=test_size,
                                         input_size=input_size, 
                                         n_workers=n_workers,
                                         n_firms=n_firms,
                                         matcher_class=matcher_class)
    
    # model
    print("prepare model")
    model_path = args.model_path
    model = NeuralSD(input_size=input_size)
    model.load_state_dict(torch.load(model_path))
    model.to('cpu')
    model.eval()

    # prediction of post-training
    print("compute prediction")
    ranking_nsd = model.predict_ranking([data[0] for data in test_batch], how='scatter')
    
    # prediction of rsd
    ranking_rsd = []
    for data in test_batch:
        worker_preferences = data[0][0]
        firm_preferences = data[0][1]
        n, m = len(worker_preferences), len(firm_preferences)
        perm = np.random.permutation(n+m).tolist()
        matrix = torch.zeros((n+m, n+m), dtype=torch.float32)
        for i, p in enumerate(perm):
            matrix[i, p - 1] = 1
        ranking_rsd.append(matrix)

    # results
    print("compute results")
    _, optimal_rankings = compute_optimal_sd(test_batch=test_batch)
    number_of_correct_rankings = dict(Counter([len(optimal_rankings[i]) for i in range(len(optimal_rankings))]))
    inclusion_nsd = inclusion_rate(test_batch=test_batch, optimal_rankings=optimal_rankings, predicted_rankings=ranking_nsd)
    inclusion_rsd = inclusion_rate(test_batch=test_batch, optimal_rankings=optimal_rankings, predicted_rankings=ranking_rsd)
    records = []
    for key, val in number_of_correct_rankings.items():
        records.append((key, val, inclusion_nsd[key], inclusion_rsd[key]))
    df_inclusion = pd.DataFrame.from_records(records, columns=['num_corrects', 'num_instances', 'nsd', 'rsd'])
    df_inclusion = df_inclusion.sort_values(by='num_corrects')
    
    print("Predictions:")
    print(df_inclusion.to_string())
    
    print("Accuracy:")
    accuracy = (df_inclusion[['nsd', 'rsd']].sum() / test_size).to_dict()
    pprint.pp(accuracy)
    
    if args.save:
        filename = datetime.datetime.today().strftime('%Y_%m_%d--%H_%M_%S')
    
        with open(f'tests/test_{filename}.txt', 'w') as f:
            pp = pprint.PrettyPrinter(indent=4, stream=f)
            pp.pprint(vars(args))
    
        with open(f'tests/test_{filename}.pkl', 'wb') as f:
            dump_results = {'acc': accuracy}
            pickle.dump(dump_results, f)
            
if __name__=='__main__':
    args = ArgumentParser()
    args.add_argument('--seed',
                      type=int,
                      default=42,
                      help='random seed.')
    args.add_argument('--model_path',
                      type=str,
                      required=True,
                      help='path to model.')
    args.add_argument('--n_workers', 
                      type=int, 
                      default=10,
                      help='Number of workers. (default: 10)')
    args.add_argument('--n_firms', 
                      type=int, 
                      default=10,
                      help='Number of firms. (default: 10)')
    args.add_argument('--test_size',
                      type=int,
                      default=1000,
                      help='# of test data.')
    args.add_argument('--input_size',
                      type=int,
                      default=10,
                      help='dimension of feature vector.')
    args.add_argument('--device',
                      choices=['cpu', 'cuda'],
                      default='cpu',
                      help='Device of model and data.')
    args.add_argument('--matcher',
                      choices=['DA', 'EH', 'MH'],
                      default='DA',
                      help='Matching model. (default: DA)')
    args.add_argument('--save',
                      action='store_true',
                      help='Whether to save the result.')
    main(args.parse_args())
