import * as tf from '@tensorflow/tfjs';

export type ActivationIdentifier =
	| 'elu'
	| 'hardSigmoid'
	| 'linear'
	| 'relu'
	| 'relu6'
	| 'selu'
	| 'sigmoid'
	| 'softmax'
	| 'softplus'
	| 'softsign'
	| 'tanh'
	| 'swish'
	| 'mish'
	| 'gelu'
	| 'gelu_new';

export interface Layer {
	type: string;
}

export type DenseLayer = Layer & {
	type: 'dense';
	/** How many neurons there will be in this layer */
	units: number;
	inputShape?: number[];
	activation?: ActivationIdentifier;
};

export type SequentialModel = {
	layers: Layer[];
	loss: string;
	optimizer: string;
};

export const layerToTF = (layer: Layer): tf.layers.Layer => {
	switch (layer.type) {
		case 'dense': {
			const denseLayer = layer as DenseLayer;
			return tf.layers.dense({
				units: denseLayer.units,
				inputShape: denseLayer.inputShape,
				activation: denseLayer.activation
			});
		}
		// Add more cases for other layer types as needed
		default:
			throw new Error(`Unsupported layer type: ${layer.type}`);
	}
};

export const createTFModel = (model: SequentialModel): tf.Sequential => {
	const tfModel = tf.sequential();

	model.layers.forEach((layer) => {
		tfModel.add(layerToTF(layer));
	});

	tfModel.compile({
		loss: model.loss,
		optimizer: model.optimizer
	});

	return tfModel;
};
