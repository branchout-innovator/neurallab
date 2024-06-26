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
	let epochs = 500;

	const model = writable<SequentialModel>({
		layers: [
			{
				type: 'dense',
				activation: selectedActivation.value as ActivationIdentifier,
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
		$model.layers = [
			...$model.layers,
			{
				type: 'dense',
				activation: selectedActivation.value as ActivationIdentifier,
				units: 1,
				inputShape: [lastLayer?.units ?? 1]
			} as DenseLayer
		];
	};

	const removeLayer = () => {
		$model.layers = [...$model.layers.slice(0, $model.layers.length - 1)];
	};

	const trainModel = () => {
		// Generate some synthetic data for training.
		const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
		const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

		const tfmodel = createTFModel($model);
		// Train the model using the data.
		tfmodel
			.fit(xs, ys, { epochs })
			.then(() => {
				// Use the model to do inference on a data point the model hasn't seen before:
				console.log(tfmodel.predict(tf.tensor2d([5], [1, 1])).toString());
				// Open the browser devtools to see the output
				toast.success(`Training complete! Open the browser devtools (F12) to see the output.`);
			})
			.catch((e) => {
				toast.error(`Error while training: ${e}. Make sure the last layer has only 1 node.`);
			});
		toast.loading(`Training for ${epochs} epochs...`);
	};

	$: {
		for (const layer of $model.layers) {
			if (layer.type === 'dense') {
				(layer as DenseLayer).activation = selectedActivation.value;
			}
		}
	}

	function getWeightsBetweenLayers(model: SequentialModel, layerIndex: number): tf.Tensor {
		const tfModel = createTFModel(model);
		const weights = tfModel.layers[layerIndex].getWeights();
		return weights[0]; // Assuming the first tensor in weights array is the weight matrix
	}
	let canvasWidth = 150;

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
					{@const weights = getWeightsBetweenLayers($model, i + 1)}
					<ConnectionsVis {leftLayerHeights} {rightLayerHeights} {canvasWidth} {weights} />
				{/if}
			{/each}
		</div>
	</div>
</div>
