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
	optimizer: string | tf.Optimizer;
};

export const layerToTF = (layer: Layer): tf.layers.Layer => {
	switch (layer.type) {
		case 'dense': {
			const denseLayer = layer as DenseLayer;
			return tf.layers.dense({
				units: denseLayer.units,
				inputShape: denseLayer.inputShape,
				activation: denseLayer.activation,
				kernelInitializer: 'glorotUniform'
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

export const loadUploadedCsv = async (csvFile: Blob | MediaSource, labels: string[]) => {
	const url = URL.createObjectURL(csvFile);
	const columnConfigs: {
		[key: string]: tf.data.ColumnConfig;
	} = {};
	for (const label of labels) {
		columnConfigs[label] = { isLabel: true };
	}

	const csvDataset = tf.data.csv(url, {
		columnConfigs
	});

	const numOfFeatures = (await csvDataset.columnNames()).length - 1;

	// Prepare the Dataset for training.
	const flattenedDataset = csvDataset
		.map(({ xs, ys }) => {
			// Convert xs(features) and ys(labels) from object form (keyed by
			// column name) to array form.
			return { xs: Object.values(xs), ys: Object.values(ys) };
		})
		.batch(32);

	const it = await flattenedDataset.iterator();
	const xs = [];
	const ys = [];
	// read only the data for the first 5 rows
	// all the data need not to be read once
	// since it will consume a lot of memory
	for (let i = 0; i < 5; i++) {
		const e = await it.next();
		xs.push(e.value.xs);
		ys.push(e.value.ys);
	}
	const featuresTensor = tf.tensor(xs);
	const labelsTensor = tf.tensor(ys);

	console.log(featuresTensor.shape);
	console.log(labelsTensor.shape);

	return flattenedDataset;
};
