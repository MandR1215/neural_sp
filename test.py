import torch
import torch.nn as nn
from torch.optim import Adam
from neuralsd import NeuralSD
from preference_generator import Euclidean
from utils import preference_to_matrix, matrix_to_preference, number_of_blocking_pairs, hamming_distance
import numpy as np
from algorithm import MatchingAlgorithm, DeferredAcceptance, RankHungarian, EqualWeightedHungarian, MinorityHungarian, SerialDictatorship
from scipy import stats
import random
from argparse import ArgumentParser
from utils import preference_to_rational
from loss_functions import compute_st_unit, compute_ir_unit
import pprint
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
    
def hamming_distances(test_batch, predictions):
    hamm = []
    for data, prediction in zip(test_batch, predictions):
        hamm.append(hamming_distance(data[1].numpy(), prediction.detach().numpy()))
        
    return hamm
    
def blocking_pairs(test_batch, predictions):
    bp = []
    for i, data in enumerate(test_batch):
        worker_preferences = data[0][0]
        firm_preferences = data[0][1]

        matching = predictions[i].detach().numpy()
        bp.append(number_of_blocking_pairs(worker_preferences=worker_preferences, 
                                           firm_preferences=firm_preferences, 
                                           matching=matching))
                                               
    return bp
    
def stability_violations(test_batch, predictions):
    stv = []
    for i, data in enumerate(test_batch):
        worker_preferences = preference_to_rational(data[0][0])
        firm_preferences = preference_to_rational(data[0][1]).T
        n, m = len(worker_preferences), len(firm_preferences[0])

        matching = predictions[i].detach().float()
        stv.append(compute_st_unit(r=matching[:n,:m], p=worker_preferences, q=firm_preferences))

    return stv

def reward_ratios(test_batch, predictions, matchers):
    reward_ratio = []
    for i, data in enumerate(test_batch):
        worker_preferences = data[0][0]
        firm_preferences = data[0][1]
        
        optimal_reward = matchers[i].compute_reward(worker_preferences, firm_preferences, matching=data[1].numpy())
        predict_reward = matchers[i].compute_reward(worker_preferences, firm_preferences, matching=predictions[i].detach().numpy())
        
        if optimal_reward > 0:
            reward_ratio.append(predict_reward/optimal_reward)
        else:
            reward_ratio.append(1.)
        
    return reward_ratio

def ir_violations(test_batch, predictions):
    irv = []
    for i, data in enumerate(test_batch):
        worker_preferences = preference_to_rational(data[0][0])
        firm_preferences = preference_to_rational(data[0][1]).T
        n, m = len(worker_preferences), len(firm_preferences[0])

        matching = predictions[i].detach().float()
        irv.append(compute_ir_unit(r=matching[:n,:m], p=worker_preferences, q=firm_preferences))

    return irv
    
def safe_wilcoxon(x, y, alternative='greater'):
    return "N/A" if all(x[i] == y[i] for i in range(len(x))) else stats.wilcoxon(x, y, alternative=alternative)
        
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
    test_batch, matchers = create_example_batch(n_data=test_size,
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

    # prediction 
    print("compute prediction")
    n = 4
    batches = [test_batch[i:i+n] for i in range(0, len(test_batch), n)]
    prediction_nsd = []
    for batch in tqdm(batches):
        prediction_nsd += model.predict([data[0] for data in batch], how='scatter')
        
    # prediction of rsd
    prediction_rsd = []
    for data in test_batch:
        worker_preferences = data[0][0]
        firm_preferences = data[0][1]
        n, m = len(worker_preferences), len(firm_preferences)

        sd = SerialDictatorship(n_workers=n, n_firms=m, ranking=list(np.random.permutation(n+m).tolist()))
        matching = sd.match(worker_preferences=worker_preferences, firm_preferences=firm_preferences)
        prediction_rsd.append(torch.tensor(matching))

    # results
    print("compute results")
    ## compare hamming distance
    print("compute hamming distances")
    hamm_nsd = hamming_distances(test_batch=test_batch, predictions=prediction_nsd)
    hamm_rsd = hamming_distances(test_batch=test_batch, predictions=prediction_rsd)
    
    
    dump_results = {
        'hamm_nsd' : np.asarray(hamm_nsd),
        'hamm_rsd' : np.asarray(hamm_rsd),
    }
    results = {
            'ci(hamm_nsd)'               : f"{np.mean(hamm_nsd):.3f} ± {np.std(hamm_nsd):.3f}",
            'ci(hamm_rsd)'               : f"{np.mean(hamm_rsd):.3f} ± {np.std(hamm_rsd):.3f}",
            'wilcoxon(hamm_rsd,hamm_nsd)': safe_wilcoxon(hamm_rsd, hamm_nsd),
    }
    
    if args.matcher == 'DA':
        ## compare blocking pairs
        print("compute blocking pairs")
        bp_nsd = blocking_pairs(test_batch=test_batch, predictions=prediction_nsd)

        ## compare stability violation
        print("compute stability violations")
        stv_nsd = stability_violations(test_batch=test_batch, predictions=prediction_nsd)
        
        ## compare ir violation
        print("compute ir violations")
        irv_nsd = ir_violations(test_batch=test_batch, predictions=prediction_nsd)
        
        ## compare by rsd
        print("compute performances of rsd")
        bp_rsd = blocking_pairs(test_batch=test_batch, predictions=prediction_rsd)
        stv_rsd = stability_violations(test_batch=test_batch, predictions=prediction_rsd)
        irv_rsd = ir_violations(test_batch=test_batch, predictions=prediction_rsd)
        
        results.update({
            'ci(bp_nsd)'                 : f"{np.mean(bp_nsd):.3f} ± {np.std(bp_nsd):.3f}",
            'ci(stv_nsd)'                : f"{np.mean(stv_nsd):.5f} ± {np.std(stv_nsd):.5f}",
            'ci(irv_nsd)'                : f"{np.mean(irv_nsd):.5f} ±  {np.std(irv_nsd):.5f}",
            'ci(bp_rsd)'                 : f"{np.mean(bp_rsd):.3f} ± {np.std(bp_rsd):.3f}",
            'ci(stv_rsd)'                : f"{np.mean(stv_rsd):.5f} ± {np.std(stv_rsd):.5f}",
            'ci(irv_rsd)'                : f"{np.mean(irv_rsd):.5f} ±  {np.std(irv_rsd):.5f}",
            'wilcoxon(bp_rsd,bp_nsd)'    : safe_wilcoxon(bp_rsd, bp_nsd),
            'wilcoxon(stv_rsd,stv_nsd)'  : safe_wilcoxon(stv_rsd, stv_nsd),
            'wilcoxon(irv_rsd,irv_nsd)'  : safe_wilcoxon(irv_rsd, irv_nsd),
        })
        
        dump_results.update({
            'bp_nsd'  : np.asarray(bp_nsd),
            'bp_rsd'  : np.asarray(bp_rsd),
            'stv_nsd' : np.asarray(stv_nsd),
            'stv_rsd' : np.asarray(stv_rsd),  
            'irv_nsd' : np.asarray(irv_nsd),
            'irv_rsd' : np.asarray(irv_rsd)
        })
    elif args.matcher in ['EH', 'MH']:
        ## compare reward ratio
        print("compute reward ratios")
        rr_nsd = reward_ratios(test_batch=test_batch, predictions=prediction_nsd, matchers=matchers)
        
        ## compare by rsd
        print("compute performances of rsd")
        rr_rsd = reward_ratios(test_batch=test_batch, predictions=prediction_rsd, matchers=matchers)
        
        results.update({
            'ci(rr_nsd)'                 : f"{np.mean(rr_nsd):.3f} ± {np.std(rr_nsd):.3f}",
            'ci(rr_rsd)'                 : f"{np.mean(rr_rsd):.3f} ± {np.std(rr_rsd):.3f}",
            'wilcoxon(rr_rsd,rr_nsd)'    : safe_wilcoxon(rr_rsd, rr_nsd, 'less'),
        })
        
        dump_results.update({
            'rr_nsd'  : np.asarray(rr_nsd),
            'rr_rsd'  : np.asarray(rr_rsd), 
        })
        
    pprint.pp(dict((key, results[key]) for key in sorted(results.keys())))
    
    if args.save:
        filename = datetime.datetime.today().strftime('%Y_%m_%d--%H_%M_%S')
    
        with open(f'tests/test_{filename}.txt', 'w') as f:
            pp = pprint.PrettyPrinter(indent=4, stream=f)
            pp.pprint(vars(args))
            pp.pprint(results)
    
        with open(f'tests/test_{filename}.pkl', 'wb') as f:
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
