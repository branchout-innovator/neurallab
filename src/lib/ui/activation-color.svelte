<script lang="ts">
	import { getSampledOutputForNode, type SampledOutputs } from '$lib/structures';
	import { getContext, onMount } from 'svelte';
	import type { Writable } from 'svelte/store';
	import * as d3 from 'd3';
	import type { HTMLAttributes } from 'svelte/elements';
	import * as tf from '@tensorflow/tfjs';
	type $$Props = HTMLAttributes<HTMLDivElement> & {
		nodeIndex: number;
		layerName: string;
		customDensity?: number;
	};

	export let nodeIndex: number;
	export let layerName: string;

	const sampledOutputs: Writable<SampledOutputs<number>> = getContext('sampledOutputs');
	const getTfModel = getContext('getTfModel') as () => tf.Sequential;
	let tfModel = getTfModel();

	let color = '#ffffff00';

	$: activationVal = $sampledOutputs[layerName] && $sampledOutputs[layerName].values[nodeIndex];

	function updateColor() {
		if (!$sampledOutputs[layerName]) return;
		const colorScale = d3
			.scaleSequential(d3.interpolateRdBu)
			.domain([d3.max($sampledOutputs[layerName].values) ?? 1, -1]);

		color = d3.rgb(colorScale($sampledOutputs[layerName].values[nodeIndex])).toString();
	}

	$: {
		$sampledOutputs;
		updateColor();
	}
</script>

<div
	style={`background-color: ${color};`}
	{...$$restProps}
	title={`Activation: ${activationVal}`}
></div>
