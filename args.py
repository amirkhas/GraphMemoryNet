import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='GMN arguments')

    parser.add_argument('--datadir', dest='datadir', type=str, default='data', help='Benchmark directory')
    parser.add_argument('--logdir', dest='logdir', type=str, default='log', help='Checkpoint and summary directory')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Dataset name')

    parser.add_argument('--cuda_index', dest='cuda_index', type=str, default='1', help='GPU index')
    parser.add_argument('--cuda', dest='cuda', type=bool, default=True, help='Use GPU if set to True')

    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=1000, help='Maximum #nodes')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--clip', dest='clip', type=float, default=2.0, help='Gradient clipping')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=2000, help='#Epochs')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--skip_connection', dest='skip_connection', type=bool, default=True, help='Skip connection')
    parser.add_argument('--patience', dest='patience', type=int, default=200, help='Patience for early stopping')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.0, help='Optimizer weight decay')
    parser.add_argument('--decay_step', dest='decay_step', type=int, default=400, help='#Epochs to decay learning rate')
    parser.add_argument('--batchnorm', dest='batchnorm', type=bool, default=False, help='if True batchnorm')

    parser.add_argument('--num_workers', dest='num_workers', type=int, default=6, help='#Data-loader workers')
    parser.add_argument('--input-dim', dest='input_dim', type=int, help='Input feature dimension')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int, help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int, help='#Classes')
    parser.add_argument('--num_folds', dest='num_folds', type=int, default=10, help='#Folds')
    parser.add_argument('--fold_index', dest='fold_index', type=int, default=0, help='Index of the fold to run')

    parser.add_argument('--num_centroids', dest='num_centroids', type=str, default='10, 1',
                        help='Number of centroids per layer, e.g "10, 1" indicates 10 and 1 clusters in layers 1 and 2')
    parser.add_argument('--cHeadsPool', dest='cHeadsPool', type=str, default='conv',
                        help='Pooling type for cluster heads: mean, max, conv')
    parser.add_argument('--cluster_heads', dest='cluster_heads', type=int, default=5, help='#Heads for the cluster')
    parser.add_argument('--p2p', dest='p2p', type=bool, default=True,
                        help='Point-to-point clustering. "True" for memory-read operation.')
    parser.add_argument('--num_clusteriter', dest='num_clusteriter', type=int, default=1,
                        help='#Iterations for clustering')
    parser.add_argument('--learn_centroid', dest='learn_centroid', type=str, default='a',
                        help='"f": fixed centroids, "c": apply unsup. loss only to centroid, '
                             '"a": apply unsup. loss only to update both centroids and model params')
    parser.add_argument('--linear_block', dest='linear_block', type=bool, default=False,
                        help='Use linear transformation between hierarchy blocks')

    parser.add_argument('--backward_period', dest='backward_period', type=int, default=5,
                        help='Frequency of applying gradients from unsupervised loss (epochs)')
    parser.add_argument('--avg_grad', dest='avg_grad', type=bool, default=True,
                        help='Average batch gradients for unsupervised loss')

    parser.add_argument('--normalize_adj', dest='normalize_adj', type=bool, default=False, help='Normalizing adjacency')
    parser.add_argument('--positional_hiddim', dest='positional_hiddim', type=int, default=16,
                        help='Hidden dimension of the input positional embedding')
    parser.add_argument('--use_rwr', dest='use_rwr', type=bool, default=True,
                        help='If True uses RWR as positional embeddings, otherwise uses adjacency')

    return parser.parse_args()
