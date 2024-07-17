def load_mapping(path, mapping):
    if mapping.lower() == 'linear':
        from optimization.optimizer import LinearFitting
        mapping = LinearFitting.from_file(path)
    elif mapping.lower() == 'affine':
        from optimization.optimizer import AffineFitting
        mapping = AffineFitting.from_file(path)
    elif mapping.lower() == 'neuralnetwork':
        from optimization.optimizer import NeuralNetworkFitting
        mapping = NeuralNetworkFitting.from_file(path)
    elif mapping.lower() == 'decouple':
        from optimization.optimizer import DecoupleFitting
        mapping = DecoupleFitting.from_file(path)
    elif mapping.lower() == 'decoupleaffine':
        from optimization.optimizer import DecoupleFitting
        mapping = DecoupleFitting.from_file(path, "Affine")
    elif mapping.lower() == 'adaptive':
        from optimization.optimizer import AdaptiveFitting, NeuralNetworkFitting
        mlp_path = str(path).replace('Adaptive', 'NeuralNetwork').split('_')
        mlp_path = '_'.join([*mlp_path[:10], 'dropout_128', *mlp_path[10:]])
        mlp_mapping = NeuralNetworkFitting.from_file(mlp_path)
        mapping = AdaptiveFitting.from_file(path, mlp_mapping)
    elif mapping.lower() == 'hybrid':
        from optimization.optimizer import HybridFitting, LinearFitting
        linear_path = str(path).replace('Hybrid', 'Linear').split('_')
        linear_path = '_'.join([*linear_path[:10], *linear_path[12:]])
        linear_mapping = LinearFitting.from_file(linear_path)
        mapping = HybridFitting.from_file(path, linear_mapping)
    elif mapping.lower() == 'decouple':
        from optimization.optimizer import DecoupleFitting
        mapping = DecoupleFitting.from_file(path)
    else:
        raise ValueError(f"Invalid experiment name: {mapping}")
    return mapping