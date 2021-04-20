import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Running for SHIG")

    '''
    input related
    '''

    # choosing datasets
    dataset_list = ['bitcoin_alpha', 'bitcoin_otc', 'epinions', 'slashdot']
    dataset = dataset_list[0]

    if 'bitcoin' in dataset:
        edge_path = "../input/" + dataset + ".csv"
    else:
        edge_path = "../input/" + dataset + ".txt"

    parser.add_argument("--dataset",
                        type=str,
                        default=dataset)

    parser.add_argument("--edge-path",
                        nargs="?",
                        default=edge_path,
                        help="Edge list csv.")

    parser.add_argument("--features-path",
                        nargs="?",
                        default=edge_path,
                        help="Edge list csv.")

    '''
    output related
    '''

    # specify the paths to save tensorboard logs
    parser.add_argument("--log_path", type=str, default='./logs/')


    '''
    model related
    '''

    parser.add_argument("--manifolds",
                        type=str,
                        default='Hyperboloid',
                        choices=['Euclidean', 'Hyperboloid', 'PoincareBall'],
                        )

    parser.add_argument("--r",
                        type=float,
                        default=2.,
                        help="fermi-dirac decoder parameter")

    parser.add_argument("--t",
                        type=float,
                        default=1.,
                        help="fermi-dirac decoder parameter")

    parser.add_argument("--c",
                        type=float,
                        default=1.,
                        help="culvature of the hyperbolic manifold")

    parser.add_argument("--num_layers",
                        type=int,
                        default=2,
                        help='Number of layers for GNNs')

    parser.add_argument("--use_bias",
                        type=bool,
                        default=False)

    parser.add_argument("--heads",
                        type=int,
                        default=1,
                        help='Heads for attention. Default is 1.')

    parser.add_argument("--dropout",
                        type=float,
                        default=0)

    '''
    training related
    '''
    parser.add_argument("--test-size",
                        type=float,
                        default=0.2,
                        help="Test dataset size. Default is 0.2.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for sklearn, random and cuda")


    parser.add_argument("--epochs",
                        type=int,
                        default=600,
                        help="Number of training epochs. Default is 900.")

    parser.add_argument("--auto_ml",
                        type=bool,
                        default=True,
                        help="Use optuna to find best hyperparameters")

    parser.add_argument("--n_trails",
                        type=int,
                        default=100,
                        help="Number of trails to search hyperparameters")

    parser.add_argument("--metric_to_optimize",
                        type=str,
                        default='AUC',
                        choices=['AUC', 'F1'],
                        )

    parser.add_argument("--reduction-iterations",
                        type=int,
                        default=30,
                        help="Number of SVD iterations. Default is 30.")

    parser.add_argument("--reduction-dimensions",
                        type=int,
                        default=64,
                        help="Number of SVD feature extraction dimensions. Default is 64.")


    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
                        help="Learning rate. Default is 0.01.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10 ** -5,
                        help="Learning rate. Default is 10^-5.")

    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space. E.g. 32 32.")

    parser.add_argument("--spectral-features",
                        dest="spectral_features",
                        action="store_true")

    parser.add_argument("--general-features",
                        dest="spectral_features",
                        action="store_false")

    parser.add_argument("--verbose",
                        type=bool,
                        default=True,
                        help="Print test results")

    parser.set_defaults(spectral_features=True)

    parser.set_defaults(layers=[64, 64])

    return parser.parse_args()
