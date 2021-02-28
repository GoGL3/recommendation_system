import argparse


class Args(object):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--save_path', nargs='?', default='saved_model/',
                        help='Saved model path.')
    parser.add_argument('--dataset', nargs='?', default='brunch',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--num_latent', type=int, default=8,
                        help='Embedding size.')
    parse = parser.parse_args()

    params = {
        "PATH": parse.path,
        "SAVE_PATH": parse.save_path,
        "DATASET": parse.dataset,
        "EPOCHS": parse.epochs,
        "BATCH_SIZE": parse.batch_size,
        "LAYERS": parse.layers,
        "NUM_NEG": parse.num_neg,
        "LR": parse.lr,
        "LEARNER": parse.learner,
        "VERBOSE": parse.verbose,
        "OUT": parse.out,
        "NUM_LATENT": parse.num_latent,
    }

