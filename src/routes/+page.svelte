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
		type SequentialModel
	} from '$lib/structures';
	import * as tf from '@tensorflow/tfjs';
	import { onMount } from 'svelte';

	let selectedActivation = { value: 'relu', label: 'ReLU' };

	let dense1: DenseLayer;
	$: dense1 = {
		type: 'dense',
		activation: selectedActivation.value as ActivationIdentifier,
		units: 1,
		inputShape: [1]
	};

	let model: SequentialModel;
	$: model = {
		layers: [dense1],
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

	<div class="flex h-full flex-row items-center justify-center rounded-lg border p-10 text-sm">
		Neural net editor will go here
	</div>
</div>

<style>
	section {
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		flex: 0.6;
	}

	h1 {
		width: 100%;
	}

	.welcome {
		display: block;
		position: relative;
		width: 100%;
		height: 0;
		padding: 0 0 calc(100% * 495 / 2048) 0;
	}

	.welcome img {
		position: absolute;
		width: 100%;
		height: 100%;
		top: 0;
		display: block;
	}
</style>
