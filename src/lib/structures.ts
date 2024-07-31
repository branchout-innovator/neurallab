import * as tf from '@tensorflow/tfjs';
import * as d3 from 'd3';

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

export type LayerType = 'dense' | 'conv2d' | 'maxpooling' | 'flatten';

export interface Layer {
	type: string;
	inputShape?: number[];
	batchSize?: number;
}

export type DenseLayer = Layer & {
	type: 'dense';
	/** How many neurons there will be in this layer */
	units: number;
	activation?: ActivationIdentifier;
};

export type Conv2DLayer = Layer & {
	type: 'conv2d';
	/** Kernel size in each dimension */
	kernelSize: [number, number];
	/** Stride in each dimension */
	strides: [number, number];
	/** The amount of nodes */
	filters: number;
	activation?: ActivationIdentifier;
};

export type LSTMLayer = Layer & {
	type: 'lstm';
	/** Kernel size in each dimension */
	units: number;
	activation?: ActivationIdentifier;
	recurrentActivation?: ActivationIdentifier;
};

export type MaxPoolingLayer = Layer & {
	type: 'maxpooling';
	/** Pooling size in each dimension */
	poolSize: [number, number];
	/** Stride in each dimension */
	strides: [number, number];
};

export type DropoutLayer = Layer & {
	type: 'dropout';
	/** Pooling size in each dimension */
	rate: number;
	/** Stride in each dimension */

};

export type FlattenLayer = Layer & {
	type: 'flatten';
};

export type SequentialModel = {
	layers: Layer[];
	loss: string;
	optimizer: string | tf.Optimizer;
	learningRate?: number;
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
		case 'conv2d': {
			const conv2dLayer = layer as Conv2DLayer;
			return tf.layers.conv2d({
				filters: conv2dLayer.filters,
				kernelSize: conv2dLayer.kernelSize,
				strides: conv2dLayer.strides,
				inputShape: conv2dLayer.inputShape,
				activation: conv2dLayer.activation,
				batchSize: conv2dLayer.batchSize,
				kernelInitializer: 'glorotUniform'
			});
		}
		case 'maxpooling': {
			const maxPoolingLayer = layer as MaxPoolingLayer;
			return tf.layers.maxPooling2d({
				poolSize: maxPoolingLayer.poolSize,
				strides: maxPoolingLayer.strides,
				inputShape: maxPoolingLayer.inputShape,
				batchSize: maxPoolingLayer.batchSize
			});
		}
		case 'flatten': {
			const flattenLayer = layer as FlattenLayer;
			return tf.layers.flatten({
				inputShape: flattenLayer.inputShape,
				batchSize: flattenLayer.batchSize
			});
		}
		case 'lstm': {
			const lstmLayer = layer as LSTMLayer;
			return tf.layers.lstm({
				units: lstmLayer.units, 
				returnSequences: true, 
				activation: lstmLayer.activation, 
				recurrentActivation: lstmLayer.recurrentActivation,
				inputShape: lstmLayer.inputShape,
				batchSize: lstmLayer.batchSize
			});
		}
		case 'dropout': {
			const dropoutLayer = layer as DropoutLayer;
			return tf.layers.dropout({
				rate: dropoutLayer.rate
			});
		}
		default:
			throw new Error(`Unsupported layer type: ${layer.type}`);
	}
};

export const createTFModel = (model: SequentialModel): tf.Sequential => {
	const tfModel = tf.sequential();

	model.layers.forEach((layer) => {
		tfModel.add(layerToTF(layer));
	});

	const optimizer = tf.train.adam(model.learningRate);

	tfModel.compile({
		loss: model.loss,
		optimizer: tf.train.adam(model.learningRate)
	});

	return tfModel;
};

export const createRNNModel = (indim: number, batch: number, slen: number): tf.Sequential => {
	const tfModel = tf.sequential();
	tfModel.add(tf.layers.dropout({rate: 0.2, inputShape: [slen, indim], batchSize: batch, dtype: 'float32'}))
	tfModel.add(tf.layers.lstm({units: 64, returnSequences: true}))
	tfModel.add(tf.layers.dropout({rate: 0.2}))
	tfModel.add(tf.layers.flatten());
	tfModel.add(tf.layers.dense({units: indim, activation: "softmax"}));
	tfModel.compile({
		loss: "categoricalCrossentropy",
		optimizer: tf.train.adam()
	});

	return tfModel;
};

export const loadUploadedCsv = async (
	csvFile: Blob | MediaSource,
	columnConfigs: {
		[key: string]: tf.data.ColumnConfig;
	}
) => {
	const url = URL.createObjectURL(csvFile);

	return await loadCsvDataset(url, columnConfigs);
};

export async function loadCsvDataset(
	url: string,
	columnConfigs: {
		[key: string]: tf.data.ColumnConfig;
	}
) {
	const csvDataset = tf.data.csv(url, {
		columnConfigs
	});

	// const numOfFeatures = (await csvDataset.columnNames()).length - 1;
	// Prepare the Dataset for training.
	const flattenedDataset = csvDataset
		// @ts-expect-error unclear type for map
		.map(({ xs, ys }) => {
			// Convert xs(features) and ys(labels) from object form (keyed by
			// column name) to array form.
			return { xs: Object.values(xs), ys: Object.values(ys) };
		});
	// .batch(64);

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

	return { dataset: flattenedDataset, columnNames: await csvDataset.columnNames() };
}

export interface SampledOutputs<T> {
	[layerName: string]: {
		shape: number[];
		values: T[];
	};
}

export async function updateSampledOutputs(
	model: tf.LayersModel,
	DENSITY: number,
	xDomain: [number, number],
	yDomain: [number, number]
): Promise<SampledOutputs<number[][]>> {
	const outputs: SampledOutputs<number[][]> = {};

	const xScale = tf.linspace(xDomain[0], xDomain[1], DENSITY);
	const yScale = tf.linspace(yDomain[1], yDomain[0], DENSITY);

	const [xx, yy] = tf.meshgrid(xScale, yScale);
	let currentInput: tf.Tensor = tf.stack([xx.flatten(), yy.flatten()], 1);

	// Forward pass through each layer, storing intermediate outputs
	for (const layer of model.layers) {
		currentInput = layer.apply(currentInput) as tf.Tensor;

		// Reshape the output to [numNodes, DENSITY, DENSITY]
		const reshapedOutput = currentInput
			.reshape([DENSITY * DENSITY, -1])
			.transpose()
			.reshape([-1, DENSITY, DENSITY]);

		// Store the shape and values
		outputs[layer.name] = {
			shape: currentInput.shape.slice(1), // Remove batch dimension
			values: reshapedOutput.arraySync() as number[][][]
		};
	}

	return outputs;
}

export async function getSampledOutputForNode(
	model: tf.LayersModel,
	layerName: string,
	nodeIndex: number,
	DENSITY: number,
	xDomain: [number, number],
	yDomain: [number, number]
): Promise<number[][]> {
	const xScale = tf.linspace(xDomain[0], xDomain[1], DENSITY);
	const yScale = tf.linspace(yDomain[1], yDomain[0], DENSITY);

	const [xx, yy] = tf.meshgrid(xScale, yScale);
	let currentInput: tf.Tensor = tf.stack([xx.flatten(), yy.flatten()], 1);

	let targetLayer: tf.layers.Layer | undefined;
	let targetLayerIndex: number = -1;

	// Find the target layer
	for (let i = 0; i < model.layers.length; i++) {
		if (model.layers[i].name === layerName) {
			targetLayer = model.layers[i];
			targetLayerIndex = i;
			break;
		}
	}

	if (!targetLayer) {
		throw new Error(`Layer "${layerName}" not found in the model.`);
	}

	// Forward pass up to the target layer
	for (let i = 0; i <= targetLayerIndex; i++) {
		currentInput = model.layers[i].apply(currentInput) as tf.Tensor;
	}

	// Reshape the output to [DENSITY * DENSITY, numNodes]
	const reshapedOutput = currentInput.reshape([DENSITY * DENSITY, -1]);

	// Select the specific node's output and reshape to [DENSITY, DENSITY]
	const nodeOutput = reshapedOutput.slice([0, nodeIndex], [-1, 1]).reshape([DENSITY, DENSITY]);

	return nodeOutput.arraySync() as number[][];
}

export async function updateSampledOutputs1D(
	model: tf.LayersModel,
	DENSITY: number,
	xDomain: [number, number]
): Promise<SampledOutputs<number[]>> {
	const outputs: SampledOutputs<number[]> = {};

	const xScale = tf.linspace(xDomain[0], xDomain[1], DENSITY);

	let currentInput: tf.Tensor = xScale.reshape([-1, 1]);

	// Forward pass through each layer, storing intermediate outputs
	for (const layer of model.layers) {
		currentInput = layer.apply(currentInput) as tf.Tensor;

		// Reshape the output to [numNodes, DENSITY]
		const reshapedOutput = currentInput.transpose();

		// Store the shape and values
		outputs[layer.name] = {
			shape: currentInput.shape.slice(1), // Remove batch dimension
			values: reshapedOutput.arraySync() as number[][]
		};
	}

	return outputs;
}

export async function getSampledOutputForNode1D(
	model: tf.LayersModel,
	layerName: string,
	nodeIndex: number,
	DENSITY: number,
	xDomain: [number, number]
): Promise<number[]> {
	const xScale = tf.linspace(xDomain[0], xDomain[1], DENSITY);

	let currentInput: tf.Tensor = xScale.reshape([-1, 1]);

	let targetLayer: tf.layers.Layer | undefined;
	let targetLayerIndex: number = -1;

	// Find the target layer
	for (let i = 0; i < model.layers.length; i++) {
		if (model.layers[i].name === layerName) {
			targetLayer = model.layers[i];
			targetLayerIndex = i;
			break;
		}
	}

	if (!targetLayer) {
		throw new Error(`Layer "${layerName}" not found in the model.`);
	}

	// Forward pass up to the target layer
	for (let i = 0; i <= targetLayerIndex; i++) {
		currentInput = model.layers[i].apply(currentInput) as tf.Tensor;
	}

	// Select the specific node's output
	const nodeOutput = currentInput.slice([0, nodeIndex], [-1, 1]);

	return nodeOutput.squeeze().arraySync() as number[];
}

export type NestedArray = number | NestedArray[];

function removeFirstDimension(arr: NestedArray): NestedArray {
	return Array.isArray(arr) ? arr[0] : arr;
}

export async function updateSampledOutputsSingle(
	model: tf.LayersModel,
	input: tf.TensorContainer
): Promise<SampledOutputs<NestedArray>> {
	const outputs: SampledOutputs<NestedArray> = {};

	// Convert input to tensor if it's not already
	let inputTensor = input instanceof tf.Tensor ? input : tf.tensor(input as tf.TensorLike);

	// Ensure input has a batch dimension
	if (inputTensor.shape[0] !== 1) {
		inputTensor = inputTensor.expandDims(0);
	}

	let currentInput: tf.Tensor = inputTensor;

	// Forward pass through each layer, storing intermediate outputs
	for (const layer of model.layers) {
		currentInput = layer.apply(currentInput) as tf.Tensor;

		// Get the activation values for this layer
		const activations = (await currentInput.array()) as NestedArray;

		// Store the shape and values
		outputs[layer.name] = {
			shape: currentInput.shape.slice(1), // Remove batch dimension
			values: removeFirstDimension(activations) as NestedArray[] // Remove batch dimension
		};
	}

	// Clean up tensors
	tf.dispose([inputTensor, currentInput]);

	return outputs;
}

// Helper function to construct input (if needed)
function constructInput(x: number, y: number): tf.Tensor {
	return tf.tensor([[x, y]]);
}
