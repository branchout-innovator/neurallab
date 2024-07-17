<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
	import type { DenseLayer, SequentialModel } from '$lib/structures';
	import { remToPx } from '$lib/utils';
	import Minus from 'lucide-svelte/icons/minus';
	import Plus from 'lucide-svelte/icons/plus';
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import * as tf from '@tensorflow/tfjs';
	import { browser } from '$app/environment';
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';
	import Heatmap from './heatmap.svelte';
	import * as HoverCard from '$lib/components/ui/hover-card';
	import EnlargedHeatmap from './EnlargedHeatmap.svelte';
	import PredictionCurve from './prediction-curve.svelte';
	import isEqual from 'lodash.isequal';

	export let layer: DenseLayer;
	export let index: number;
	export let tfLayer: tf.layers.Layer;

	const model: Writable<SequentialModel> = getContext('model');

	const setUnits = (units: number) => {
		($model.layers[index] as DenseLayer).units = units;
		// const nextLayer = $model.layers[index + 1] as DenseLayer;
		// if (nextLayer) {
		// 	nextLayer.inputShape = [layer.units];
		// }
	};

	$: biases = tfLayer.getWeights()[1];

	let nodes: { bias: number; normalizedBias: number }[] = [];

	function getMaxAbsWeight(
		arr:
			| number
			| number[]
			| number[][]
			| number[][][]
			| number[][][][]
			| number[][][][][]
			| number[][][][][][]
	): number {
		if (typeof arr === 'number') {
			return Math.abs(arr);
		}
		return Math.max(...arr.map(getMaxAbsWeight));
	}

	const updateNodes = async (biases: tf.Tensor) => {
		if (!browser) return;
		const biasesArray = (await biases.array()) as number[];
		if (typeof biasesArray === 'number') return;
		const maxBias = getMaxAbsWeight(biasesArray);

		nodes = [];
		for (let i = 0; i < biasesArray.length; i++) {
			const bias = biasesArray[i];
			const normalizedBias = bias / maxBias;
			nodes.push({ bias, normalizedBias });
		}
	};

	$: {
		updateNodes(biases);
	}

	function getColor(normalizedWeight: number): string {
		return normalizedWeight > 0 ? '#EF4444' : '#3B82F6';
	}

	$: is1D = isEqual($model.layers[0].inputShape, [1]);
	$: is2D = isEqual($model.layers[0].inputShape, [2]);

	$: xDomain = (is1D ? [-11, 11] : [-3, 3]) as [number, number];
	$: yDomain = (is1D ? [-10, 120] : [-3, 3]) as [number, number];
</script>

<div class="flex flex-col items-center gap-2 rounded-lg border bg-card p-2 text-card-foreground">
	<Button variant="ghost" size="icon" class="h-6 w-6" on:click={() => setUnits(layer.units + 1)}>
		<Plus class="h-4 w-4"></Plus>
	</Button>
	<Button
		variant="ghost"
		size="icon"
		class="h-6 w-6"
		on:click={() => setUnits(Math.max(layer.units - 1, 1))}
	>
		<Minus class="h-4 w-4"></Minus>
	</Button>
	{#each { length: layer.units } as _, nodeIndex (nodeIndex)}
		<div class="relative flex h-6 w-6 items-center justify-center">
			<HoverCard.Root>
				<HoverCard.Trigger>
					{#if isEqual($model.layers[0].inputShape, [1])}
						<PredictionCurve
							{nodeIndex}
							layerName={tfLayer.name}
							class="h-5 w-5 rounded-[0.15rem]"
							{xDomain}
							{yDomain}
						/>
					{:else if isEqual($model.layers[0].inputShape, [2])}
						<Heatmap
							{nodeIndex}
							layerName={tfLayer.name}
							class="rounded-[0.15rem]"
							{xDomain}
							{yDomain}
						/>
					{/if}
				</HoverCard.Trigger>
				<HoverCard.Content class="h-fit max-h-none w-fit max-w-none">
					<EnlargedHeatmap {nodeIndex} layerName={tfLayer.name} {xDomain} {yDomain} />
				</HoverCard.Content>
			</HoverCard.Root>

			<Tooltip.Root>
				<Tooltip.Trigger
					class="absolute left-[-0.3rem] h-1 w-1 rounded-full"
					style={`background-color: ${getColor(nodes[nodeIndex]?.normalizedBias)}; opacity: ${Math.abs(nodes[nodeIndex]?.normalizedBias)};`}
					aria-label={`Bias: ${nodes[nodeIndex]?.bias}`}
				></Tooltip.Trigger>
				<Tooltip.Content>
					Bias: {nodes[nodeIndex]?.bias}
				</Tooltip.Content>
			</Tooltip.Root>
		</div>
	{/each}
</div>
