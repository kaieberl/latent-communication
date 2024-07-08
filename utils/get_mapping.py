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
    elif mapping.lower() == 'adaptive':
        from optimization.optimizer import AdaptiveFitting
        mapping = AdaptiveFitting.from_file(path)
    else:
        raise ValueError(f"Invalid experiment name: {mapping}")
    return mapping