<script lang="ts">
	import type { DenseLayer, SampledOutputs } from '$lib/structures';
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import * as tf from '@tensorflow/tfjs';

	export let columnNames: string[];
	export let currentExample: { xs: number[]; ys: number[] } | null;
	export let tfLayer: tf.layers.Layer;
	export let isImageDataset = false;
	export let labellist: string[] = [];

	const csvColumnConfigs: Writable<{
		[key: string]: { isLabel: 'true' | 'false' };
	}> = getContext('csvColumnConfigs');

	$: inputFeatures = columnNames
		.filter((c) => $csvColumnConfigs[c]?.isLabel === 'true')
		.map((c, i) => ({ name: c, value: currentExample?.ys[i] }));

	const sampledOutputs: Writable<SampledOutputs<number>> = getContext('sampledOutputs');

	const countDecimals = (x: number) => {
		if (Math.floor(x.valueOf()) === x.valueOf()) return 0;
		return x.toString().split('.')[1].length || 0;
	};
</script>

<div class="flex flex-col items-start gap-2 rounded-lg py-2 text-card-foreground">
	<h5 class="mb-9 text-sm">&nbsp;</h5>
	{#if isImageDataset}
		{#each labellist as feature}
			<div class="flex flex-row items-center">
				<div class="w-6 border-b border-muted-foreground"></div>
				<div
				class="flex flex-row items-start justify-center text-nowrap rounded bg-muted px-2 py-1 font-medium"
			>
					<span class="text-xs">{feature}</span>
				</div>
			</div>
		{/each}
	{:else}
	{#each inputFeatures as feature, nodeIndex (nodeIndex)}
		{@const predictionVal =
			$sampledOutputs[tfLayer.name] && $sampledOutputs[tfLayer.name].values[nodeIndex]}
		<div class="flex flex-row items-center">
			<div class="w-6 border-b border-muted-foreground"></div>
			<div
				class="flex flex-row items-start justify-center text-nowrap rounded bg-muted px-2 py-1 font-medium"
			>
				{#if feature.value && predictionVal}
					<span class="text-xs"
						>Predicted {feature.name}: {predictionVal.toFixed(countDecimals(feature.value))}</span
					>
				{:else}
					<span class="text-xs">{feature.name}</span>
				{/if}
			</div>
			<div class="ml-2 text-nowrap">
				{#if feature.value}
					<span class="text-xs">Actual {feature.name}: {feature.value}</span>
				{/if}
			</div>
		</div>
	{/each}
	{/if}
</div>
