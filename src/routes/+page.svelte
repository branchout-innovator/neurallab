<script lang="ts">
	import Counter from './Counter.svelte';
	import welcome from '$lib/images/svelte-welcome.webp';
	import welcome_fallback from '$lib/images/svelte-welcome.png';

	import * as Select from '$lib/components/ui/select';
	import Label from '$lib/components/ui/label/label.svelte';
	import {
		createTFModel,
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

	const layerComponents: Record<string, typeof SvelteComponent> = {
		dense: DenseLayerVis as typeof SvelteComponent
	};

	let selectedActivation = { value: 'relu' as ActivationIdentifier, label: 'ReLU' };
	let epochs = 50;

	const model = writable<SequentialModel>({
		layers: [
			{
				type: 'dense',
				units: 1,
				inputShape: [1]
			} as DenseLayer
		],
		loss: 'meanSquaredError',
		optimizer: 'sgd'
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
				units: 1,
				inputShape: [lastLayer?.units ?? 1]
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

	const trainModel = async () => {
		function generateData(numPoints: number, range: { max: number; min: number }) {
			const xs = [];
			const ys = [];
			for (let i = 0; i < numPoints; i++) {
				const x = Math.random() * (range.max - range.min) + range.min;
				xs.push(x);
				ys.push(x * x);
			}
			return { xs: tf.tensor2d(xs, [numPoints, 1]), ys: tf.tensor2d(ys, [numPoints, 1]) };
		}

		// Generate some synthetic data for training.
		const { xs, ys } = generateData(100, { min: -10, max: 10 });

		toast.loading(`Training for ${epochs} epochs...`);

		// Train the model using the data.
		await tfModel.fit(xs, ys, { epochs }).catch((e) => {
			toast.error(`Error while training: ${e}. Make sure the last layer has only 1 node.`);
		});
		console.log(tfModel);
		console.log($model);

		// Use the model to do inference on a data point the model hasn't seen before:
		console.log(tfModel.predict(tf.tensor2d([2], [1, 1])).toString());
		console.log(tfModel.predict(tf.tensor2d([-2], [1, 1])).toString());
		// Open the browser devtools to see the output
		toast.success(`Training complete! Open the browser devtools (F12) to see the output.`);
		tfModel = tfModel;
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

	// to draw weight connections: https://github.com/tensorflow/playground/blob/02469bd3751764b20486015d4202b792af5362a6/src/playground.ts#L538
</script>

<svelte:head>
	<title>NeuralLab</title>
	<meta name="description" content="Design and visualize neural networks in your browser." />
</svelte:head>

<div class="container flex h-full max-w-screen-2xl flex-col gap-4 py-4">
	<div class="flex flex-row items-end gap-4">
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
			<Input type="number" bind:value={epochs} placeholder="1000" min={1} />
		</div>
		<div class="flex-1"></div>
		<Button on:click={trainModel}>
			<Brain class="mr-2 h-4 w-4"></Brain>
			Train
		</Button>
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
				<svelte:component this={layerComponents[layer.type]} {layer} index={i}></svelte:component>
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
