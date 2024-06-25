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
	import { onMount, SvelteComponent } from 'svelte';
	import DenseLayerVis from '$lib/ui/dense-layer-vis.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import Plus from 'lucide-svelte/icons/plus';
	import Minus from 'lucide-svelte/icons/minus';

	const layerComponents: Record<string, typeof SvelteComponent> = {
		dense: DenseLayerVis as typeof SvelteComponent
	};

	let selectedActivation = { value: 'relu', label: 'ReLU' };

	let model: SequentialModel;
	$: model = {
		layers: [
			{
				type: 'dense',
				activation: selectedActivation.value as ActivationIdentifier,
				units: 1,
				inputShape: [1]
			}
		],
		loss: 'meanSquaredError',
		optimizer: 'sgd'
	};

	onMount(() => {
		// Generate some synthetic data for training.
		const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
		const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

		const tfmodel = createTFModel(model);
		// Train the model using the data.
		tfmodel.fit(xs, ys, { epochs: 10 }).then(() => {
			// Use the model to do inference on a data point the model hasn't seen before:
			console.log(tfmodel.predict(tf.tensor2d([5], [1, 1])).toString());
			// Open the browser devtools to see the output
		});
	});

	const addLayer = () => {
		model.layers = [
			...model.layers,
			{
				type: 'dense',
				activation: selectedActivation.value as ActivationIdentifier,
				units: 1,
				inputShape: [1]
			} as DenseLayer
		];
	};

	const removeLayer = () => {
		model.layers = [...model.layers.slice(0, model.layers.length - 1)];
	};
</script>

<svelte:head>
	<title>Home</title>
	<meta name="description" content="Svelte demo app" />
</svelte:head>

<div class="container flex h-full max-w-screen-2xl flex-col gap-4 py-4">
	<div class="flex flex-col gap-2">
		<Label class="text-xs">Activation Function</Label>
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

	<div class="flex h-full flex-col items-center justify-center gap-6 rounded-lg border p-6 text-sm">
		<div class="flex flex-row items-center">
			<Button variant="ghost" size="icon" class="h-8 w-8" on:click={addLayer}>
				<Plus class="h-4 w-4"></Plus>
			</Button>
			<Button variant="ghost" size="icon" class="h-8 w-8" on:click={removeLayer}>
				<Minus class="h-4 w-4"></Minus>
			</Button>
			<span class="ml-2 leading-none text-muted-foreground">{model.layers.length} Layers</span>
		</div>

		<div class="flex flex-grow flex-row items-start gap-6">
			{#each model.layers as layer, i (i)}
				<svelte:component this={layerComponents[layer.type]} {layer}></svelte:component>
			{/each}
		</div>
	</div>
</div>
