import pandas as pd
import numpy as np
from scipy.stats import kruskal
from sklearn.cluster import AgglomerativeClustering
import argparse
import os
import us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-z", default=2, type=int, help="Maximum z score for a pair of servers. Default is 2")
    parser.add_argument("-v", default=0.1, type=float, help="Maximum coefficient of variation for a point of measurement between a pair of servers. Default is 0.1")
    parser.add_argument("-c", default="rtt", type=str, help="rtt or rtt_normalized. Default is rtt")
    parser.add_argument("-w", default=None, type=str, help="Path to a weights file. Default is  None")
    args = parser.parse_args()

    column = args.c

    pickle_filename = get_or_create_pickle(column=column, cov_max=args.v, z=args.z)

    print("Generating results...")

    pairs = pd.read_pickle(pickle_filename)

    if args.w is not None:
        if os.path.isfile(args.w):
            weights = pd.read_csv(args.w)
            weights['authoritative_state'] = weights['state'].map(us.states.mapping('name', 'abbr'))
            weights = weights[['authoritative_state', 'value']]
            weights.value = weights.value.astype(int)

            pairs = pairs.reset_index().merge(weights, how='inner')
            pairs['median'] = pairs['median'] * pairs['value']

            del pairs['value']

            pairs = pairs.set_index(['recursive_state', 'authoritative_state', 'recursive_ip', 'authoritative_ip'])
        else:
            args.w = None

    # Create main results directory if it doesn't exist
    if not os.path.isdir('./results'):
        os.mkdir('./results')

    # Create results directory if it doesn't exist
    if args.w is not None:
        results_dir = './results/results_{}_{}_{}_{}/'.format(args.c, args.z, args.v,  os.path.basename(args.w))
    else:
        results_dir = './results/results_{}_{}_{}_{}/'.format(args.c, args.z, args.v,  "None")
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # Generating rankings, along with the count for each state
    state_rankings = pairs.groupby(['recursive_state'])[['median']].agg([np.median, 'count'])
    state_rankings.columns = state_rankings.columns.droplevel()
    state_rankings = state_rankings.sort_values('median')
    state_rankings['rank'] = state_rankings['median'].rank(method='max')

    state_to_state_p_values = pd.DataFrame()
    state_to_state_p_values_i = pd.DataFrame()
    state_to_state_h_values = pd.DataFrame()

    # Generate kruskals values for each state pair
    for i in range(0, state_rankings.shape[0]):
        s1 = list(pd.DataFrame(state_rankings.iloc[[i]]).index)[0]

        for j in range(0, state_rankings.shape[0]):
            s2 = list(pd.DataFrame(state_rankings.iloc[[j]]).index)[0]

            h, p = kruskal(pairs.loc[[s1]]['median'], pairs.loc[[s2]]['median'])

            state_to_state_p_values.at[s1, s2] = p
            state_to_state_p_values_i.at[s1, s2] = 1 - p
            state_to_state_h_values.at[s1, s2] = h

    # Generate clusters based on inverted kruskal's values
    ac = AgglomerativeClustering(distance_threshold=.5, n_clusters=None, compute_full_tree=True).fit(state_to_state_p_values_i)
    state_rankings['cluster'] = ac.labels_

    # Save output
    state_rankings.to_csv('{}/state_rankings.csv'.format(results_dir))
    state_to_state_p_values.to_csv('{}/state_to_state_p_values.csv'.format(results_dir))
    state_to_state_h_values.to_csv('{}/state_to_state_h_values.csv'.format(results_dir))


def get_or_create_pickle(column='rtt', z=2, cov_max=0.1):
    if not os.path.isdir("./pickles"):
        os.mkdir("pickles")

    pickle_filename = './pickles/{}_{}_{}_pairs_pickle.zip'.format(column, z, cov_max)

    if not os.path.isfile(pickle_filename):
        print("Creating pickle...")

        df = pd.read_csv('final_dns_dataset.csv', usecols=['rtt', 'authoritative_ip', 'recursive_ip', 'authoritative_state',
                                                           'recursive_state', 'rtt_normalized'])

        # Generate z scores for individual measurements within server pairs
        df['z_score'] = df.groupby(['recursive_state', 'authoritative_state',
                                    'recursive_ip', 'authoritative_ip'])[column].apply(lambda x: (x - x.mean())/x.std())

        # Filter out measurements that have z scores above the limit
        df = df[abs(df['z_score']) <= z]

        # Determine the coefficient of variation for a given series
        def cov(x):
            return np.std(x) / np.mean(x)

        # Generate coefficient of variation for server pairs
        pairs = df.groupby(['recursive_state', 'authoritative_state',
                            'recursive_ip', 'authoritative_ip'])[column].agg([np.median, cov])
        # Filter out pairs with coefficient of variations too high - these are unreliable
        pairs = pairs[pairs['cov'] < cov_max]

        del pairs['cov']

        pairs.to_pickle(pickle_filename)

    return pickle_filename


if __name__ == "__main__":
    main()
