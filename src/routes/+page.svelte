<script lang="ts">
	import * as Select from '$lib/components/ui/select';
	import Label from '$lib/components/ui/label/label.svelte';
	import {
		createTFModel,
		loadUploadedCsv,
		type ActivationIdentifier,
		type DenseLayer,
		type Layer,
		type SequentialModel
	} from '$lib/structures';
	import * as tf from '@tensorflow/tfjs';
	import { onMount, setContext, SvelteComponent } from 'svelte';
	import DenseLayerVis from '$lib/ui/dense-layer-vis.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import Plus from 'lucide-svelte/icons/plus';
	import Minus from 'lucide-svelte/icons/minus';
	import Brain from 'lucide-svelte/icons/brain';
	import Activity from 'lucide-svelte/icons/activity';
	import RefreshCw from 'lucide-svelte/icons/refresh-cw';
	import { writable } from 'svelte/store';
	import Input from '$lib/components/ui/input/input.svelte';
	import { toast } from 'svelte-sonner';
	import ConnectionsVis from '$lib/ui/connections-vis.svelte';
	import { getNodeYPositions } from '$lib/ui/connections-vis';
	import { Switch } from '$lib/components/ui/switch/index.js';
	import { browser } from '$app/environment';
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';
	import FileInput from '$lib/components/ui/file-input/file-input.svelte';

	const layerComponents: Record<string, typeof SvelteComponent> = {
		dense: DenseLayerVis as typeof SvelteComponent
	};

	let selectedActivation = { value: 'relu' as ActivationIdentifier, label: 'ReLU' };
	let epochs = 1000;

	const model = writable<SequentialModel>({
		layers: [
			{
				type: 'dense',
				units: 10,
				inputShape: [1]
			} as DenseLayer,
			{
				type: 'dense',
				units: 10
			} as DenseLayer,
			{
				type: 'dense',
				units: 1
			} as DenseLayer
		],
		loss: 'meanSquaredError',
		optimizer: 'adam'
	});
	setContext('model', model);

	const addLayer = () => {
		const lastLayer = $model.layers[$model.layers.length - 1] as DenseLayer;
		// if (lastLayer.type === 'dense')
		// 	($model.layers[$model.layers.length - 1] as DenseLayer).activation =
		// 		selectedActivation.value as ActivationIdentifier;
		$model.layers = [
			...$model.layers,
			{
				type: 'dense',
				units: 1
			} as DenseLayer
		];
		console.log($model.layers);
		updateTFModel($model);
	};

	const removeLayer = () => {
		$model.layers = [...$model.layers.slice(0, $model.layers.length - 1)];
		if ($model.layers.length > 0) {
			const lastLayer = $model.layers[$model.layers.length - 1];
			if (lastLayer.type === 'dense') {
				($model.layers[$model.layers.length - 1] as DenseLayer).activation = undefined;
			}
		}
		console.log($model);
		updateTFModel($model);
	};

	let currentEpoch = 0;

	const trainModel = async () => {
		// The model needs to be "big enough" to benefit from GPU acceleration
		// So with the small models in the Tensorflow playground its actually faster to use CPU
		// await tf.setBackend('webgl');
		console.log(tf.getBackend());
		async function generateData(numPoints: number, range: { max: number; min: number }) {
			const xs = [];
			const ys = [];
			for (let i = 0; i < numPoints; i++) {
				// const x = Math.random() * (range.max - range.min) + range.min;
				const x = range.min + (i / numPoints) * (range.max - range.min);
				xs.push(x);
				ys.push(x * x);
			}
			// Normalize the data
			const xTensor = tf.tensor2d(xs, [numPoints, 1]);
			const yTensor = tf.tensor2d(ys, [numPoints, 1]);
			// const xMin = xTensor.min();
			// const xMax = xTensor.max();
			// const yMin = yTensor.min();
			// const yMax = yTensor.max();

			// const normalizedXs = xTensor.sub(xMin).div(xMax.sub(xMin));
			// const normalizedYs = yTensor.sub(yMin).div(yMax.sub(yMin));

			return { xs: await xTensor.array(), ys: await yTensor.array() };
		}

		async function generateDataset(): Promise<tf.data.Dataset<tf.TensorContainer>> {
			const { xs, ys } = await generateData(1000, { min: -10, max: 10 });
			const xDataset = tf.data.array(xs);
			const yDataset = tf.data.array(ys);
			const xyDataset = tf.data.zip({ xs: xDataset, ys: yDataset }).batch(32).shuffle(4);
			return xyDataset;
		}

		// Generate some synthetic data for training.
		const data = csvDataset || (await generateDataset());

		toast.loading(`Training for ${epochs} epochs...`);

		// Train the model using the data.
		currentEpoch = 0;

		try {
			await tfModel.fitDataset(data, {
				epochs: Number(epochs),
				callbacks: {
					onEpochEnd(epoch, logs) {
						currentEpoch = epoch + 1;
						if (currentEpoch % 5 === 0) tfModel = tfModel;
					}
				}
			});
			console.log(tfModel);
			console.log($model);

			// Make predictions and denormalize
			const input = tf.tensor2d([Number(testPred)], [1, 1]);
			// const normalizedInput = input.sub(data.xMin).div(data.xMax.sub(data.xMin));
			const prediction = tfModel.predict(input) as tf.Tensor; // Assert that this is a single tensor
			// const denormalizedPrediction = prediction.mul(data.yMax.sub(data.yMin)).add(data.yMin);
			prediction.print(); // Should print a value close to 4 (2^2)

			// Open the browser devtools to see the output
			toast.success(`Training complete! Try changing the input value and see the prediction.`);
			tfModel = tfModel;
		} catch (e) {
			console.error(e);
			toast.error(`Error while training: ${e}. Make sure the last layer has only 1 node.`, {
				duration: 100000
			});
		}
	};

	$: {
		for (let i = 0; i < $model.layers.length; i++) {
			const layer = $model.layers[i];
			if (layer.type === 'dense' && i < $model.layers.length - 1) {
				(layer as DenseLayer).activation = selectedActivation.value;
			}
		}
		updateTFModel($model);
	}

	function getWeightsBetweenLayers(model: tf.Sequential, layerIndex: number): tf.Tensor {
		const weights = model.layers[layerIndex].getWeights();
		return weights[0]; // Assuming the first tensor in weights array is the weight matrix
	}
	let canvasWidth = 150;

	let tfModel: tf.Sequential;

	const updateTFModel = (model: SequentialModel) => {
		if (!tfModel) {
			tfModel = createTFModel(model);
		} else {
			// TODO: preserve old weights when creating model with new structure
			const newModel = createTFModel(model);
			tfModel = newModel;
		}
	};

	$: {
		updateTFModel($model);
	}

	$: predictedVal = tfModel?.predict(tf.tensor2d([Number(testPred)], [1, 1]));

	let testPred = 2;

	let useGPU = false;
	$: {
		if (browser) tf.setBackend(useGPU ? 'webgl' : 'cpu');
	}

	let datasetUploadFiles: FileList;

	let csvDataset: tf.data.Dataset<tf.TensorContainer>;

	$: {
		if (datasetUploadFiles) {
			(async () => {
				csvDataset = await loadUploadedCsv(datasetUploadFiles[0], ['Squared Value']);
			})();
		}
	}
</script>

<svelte:head>
	<title>NeuralLab</title>
	<meta name="description" content="Design and visualize neural networks in your browser." />
</svelte:head>

<div class="container flex h-full max-w-screen-2xl flex-col gap-4 py-4">
	<!-- Controls (header) -->
	<div class="flex flex-row flex-wrap items-end gap-4">
		<div class="flex flex-col gap-2">
			<Label class="flex gap-2 text-xs">
				<Activity class="h-4 w-4"></Activity>
				Activation Function
			</Label>
			<Select.Root bind:selected={selectedActivation}>
				<Select.Trigger class="w-[180px]">
					<Select.Value></Select.Value>
				</Select.Trigger>
				<Select.Content>
					<Select.Item value="relu">ReLU</Select.Item>
					<Select.Item value="sigmoid">Sigmoid</Select.Item>
				</Select.Content>
			</Select.Root>
		</div>
		<div class="flex flex-col gap-2">
			<Label class="flex gap-2 text-xs">
				<RefreshCw class="h-4 w-4"></RefreshCw>
				Epochs
			</Label>
			<Input type="number" bind:value={epochs} placeholder="1000" min={1} class="w-24" />
		</div>
		<div class="flex flex-col gap-2">
			<Label class="flex gap-2 text-xs">Input</Label>
			<Input type="number" bind:value={testPred} placeholder="2" class="w-24" />
		</div>
		<div class="flex flex-col gap-2">
			<Label class="flex gap-2 text-xs">Predicted Value</Label>
			<p class="h-9 text-center text-sm leading-9">{predictedVal}</p>
		</div>
		<div class="flex flex-col gap-2">
			<Label class="flex gap-2 text-xs" for="dataset-upload">Upload Dataset</Label>
			<FileInput id="dataset-upload" class="w-32" bind:files={datasetUploadFiles} />
		</div>
		<div class="flex-1"></div>
		<div class="flex flex-col gap-2">
			<Label class="flex gap-2 text-xs">Hardware</Label>
			<Tooltip.Root>
				<Tooltip.Trigger class="flex h-9 items-center space-x-2">
					<Label for="hardware-backend">CPU</Label>
					<Switch id="hardware-backend" bind:checked={useGPU} />
					<Label for="hardware-backend">GPU</Label>
				</Tooltip.Trigger>
				<Tooltip.Content class="max-w-52">
					GPU is recommended for large models but slower for small models.
				</Tooltip.Content>
			</Tooltip.Root>
		</div>

		<div class="flex flex-col gap-2">
			<Label class="flex gap-2 text-xs">Epoch: {currentEpoch}</Label>
			<Button on:click={trainModel}>
				<Brain class="mr-2 h-4 w-4"></Brain>
				Train
			</Button>
		</div>
	</div>

	<div class="flex h-full flex-col items-center justify-center gap-6 rounded-lg border p-6 text-sm">
		<div class="flex flex-row items-center">
			<Button variant="ghost" size="icon" class="h-8 w-8" on:click={addLayer}>
				<Plus class="h-4 w-4"></Plus>
			</Button>
			<Button variant="ghost" size="icon" class="h-8 w-8" on:click={removeLayer}>
				<Minus class="h-4 w-4"></Minus>
			</Button>
			<span class="ml-2 leading-none text-muted-foreground">{$model.layers.length} Layers</span>
		</div>

		<div class="flex flex-grow flex-row items-start">
			{#each $model.layers as layer, i (i)}
				<svelte:component
					this={layerComponents[layer.type]}
					{layer}
					index={i}
					tfLayer={tfModel.layers[i]}
				></svelte:component>
				{#if i < $model.layers.length - 1}
					{@const leftLayerHeights = getNodeYPositions(layer)}
					{@const rightLayerHeights = getNodeYPositions($model.layers[i + 1])}
					{@const weights = getWeightsBetweenLayers(tfModel, i + 1)}
					<ConnectionsVis {leftLayerHeights} {rightLayerHeights} {canvasWidth} {weights} />
				{/if}
			{/each}
		</div>
	</div>
</div>
