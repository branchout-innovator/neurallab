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

export const loadUploadedCsv = async (
	csvFile: Blob | MediaSource,
	columnConfigs: {
		[key: string]: tf.data.ColumnConfig;
	}
) => {
	const url = URL.createObjectURL(csvFile);

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
		})
		.batch(64);

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

/**
 * Given a neural network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
// function updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
// 	const boundary = {};
// 	if (firstTime) {
// 		nn.forEachNode(network, true, (node) => {
// 			boundary[node.id] = new Array(DENSITY);
// 		});
// 		// Go through all predefined inputs.
// 		for (let nodeId in INPUTS) {
// 			boundary[nodeId] = new Array(DENSITY);
// 		}
// 	}
// 	let xScale = d3
// 		.scaleLinear()
// 		.domain([0, DENSITY - 1])
// 		.range(xDomain);
// 	let yScale = d3
// 		.scaleLinear()
// 		.domain([DENSITY - 1, 0])
// 		.range(xDomain);

// 	let i = 0,
// 		j = 0;
// 	for (i = 0; i < DENSITY; i++) {
// 		if (firstTime) {
// 			nn.forEachNode(network, true, (node) => {
// 				boundary[node.id][i] = new Array(DENSITY);
// 			});
// 			// Go through all predefined inputs.
// 			for (let nodeId in INPUTS) {
// 				boundary[nodeId][i] = new Array(DENSITY);
// 			}
// 		}
// 		for (j = 0; j < DENSITY; j++) {
// 			// 1 for points inside the circle, and 0 for points outside the circle.
// 			let x = xScale(i);
// 			let y = yScale(j);
// 			let input = constructInput(x, y);
// 			nn.forwardProp(network, input);
// 			nn.forEachNode(network, true, (node) => {
// 				boundary[node.id][i][j] = node.output;
// 			});
// 			if (firstTime) {
// 				// Go through all predefined inputs.
// 				for (let nodeId in INPUTS) {
// 					boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
// 				}
// 			}
// 		}
// 	}
// }

// Claude's response

export interface SampledOutputs {
	/** Key: layer.name in Tensorflow layers */
	[key: string]: number[][][];
}

interface Inputs {
	[key: string]: number[];
}

export async function updateSampledOutputs(
	model: tf.LayersModel,
	DENSITY: number,
	xDomain: [number, number]
	// INPUTS: Inputs
): Promise<SampledOutputs> {
	const outputs: SampledOutputs = {};

	model.layers.forEach((layer) => {
		outputs[layer.name] = Array(DENSITY).fill(Array(DENSITY).fill(0));
	});
	// // Go through all predefined inputs.
	// for (let inputName in INPUTS) {
	// 	outputs[inputName] = Array(DENSITY).fill(Array(DENSITY).fill(0));
	// }

	const xScale = tf.linspace(xDomain[0], xDomain[1], DENSITY);
	const yScale = tf.linspace(xDomain[1], xDomain[0], DENSITY);

	const [xx, yy] = tf.meshgrid(xScale, yScale);
	let currentInput: tf.Tensor = tf.stack([xx.flatten(), yy.flatten()], 1);

	// Forward pass through each layer, storing intermediate outputs
	for (const layer of model.layers) {
		currentInput = layer.apply(currentInput) as tf.Tensor;
		const reshapedOutput = currentInput.reshape([DENSITY, DENSITY, -1]);
		// list of nodes that each have two dimensional outputs (from the grid of inputs)
		outputs[layer.name] = reshapedOutput.arraySync() as number[][][];
	}

	// // Go through all predefined inputs.
	// for (let inputName in INPUTS) {
	// 	const inputTensor = tf.tensor(INPUTS[inputName]);
	// 	const repeatedInput = inputTensor.broadcastTo([DENSITY * DENSITY, inputTensor.shape[0]]);
	// 	outputs[inputName] = repeatedInput.reshape([DENSITY, DENSITY, -1]).arraySync() as number[][][];
	// }

	return outputs;
}

// Helper function to construct input (if needed)
function constructInput(x: number, y: number): tf.Tensor {
	return tf.tensor([[x, y]]);
}
