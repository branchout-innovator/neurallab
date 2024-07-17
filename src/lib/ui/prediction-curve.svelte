<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import * as d3 from 'd3';
	import type { HTMLAttributes } from 'svelte/elements';
	import * as tf from '@tensorflow/tfjs';
	import {
		getSampledOutputForNode,
		getSampledOutputForNode1D,
		type SampledOutputs
	} from '$lib/structures';
	import { remToPx } from '$lib/utils';

	type $$Props = HTMLAttributes<SVGElement> & {
		nodeIndex: number;
		layerName: string;
		customDensity?: number;
	};

	export let nodeIndex: number;
	export let layerName: string;
	export let customDensity: $$Props['customDensity'] = undefined;

	const sampledOutputs: Writable<SampledOutputs<number[]>> = getContext('sampledOutputs');
	const getTfModel = getContext('getTfModel') as () => tf.Sequential;
	let tfModel = getTfModel();

	let svg: SVGSVGElement;
	let gx: SVGGElement;
	let gy: SVGGElement;
	let path: SVGPathElement;

	const size = 64;

	let nodeOutputs: number[] | undefined;
	$: {
		(async () => {
			nodeOutputs = customDensity
				? await getSampledOutputForNode1D(tfModel, layerName, nodeIndex, customDensity, [-3, 3])
				: $sampledOutputs && $sampledOutputs[layerName] && $sampledOutputs[layerName][nodeIndex];
		})();
	}

	$: if (nodeOutputs) {
		updateChart();
	}

	function updateChart() {
		if (!nodeOutputs) return;
		const chartWidth = size;
		const chartHeight = size;

		const numSamples = nodeOutputs.length;
		const xScale = d3.scaleLinear().domain([-3, 3]).range([0, chartWidth]);
		const yScale = d3
			.scaleLinear()
			.domain([d3.min(nodeOutputs) || 0, d3.max(nodeOutputs) || 1])
			.range([chartHeight, 0]);

		const line = d3
			.line<number>()
			.x((_, i) =>
				xScale(
					d3
						.scaleLinear()
						.domain([0, numSamples - 1])
						.range([-3, 3])(i)
				)
			)
			.y((d) => yScale(d));

		d3.select(path).datum(nodeOutputs).attr('d', line);
	}
</script>

<svg
	bind:this={svg}
	viewBox={`0 0 ${size} ${size}`}
	class="pointer-events-none"
	overflow="visible"
	{...$$restProps}
>
	<g>
		<path bind:this={path} fill="none" stroke="#737373" stroke-width="2" />
	</g>
</svg>

<style>
	:global(.tick text) {
		font-size: 10px;
	}
</style>
