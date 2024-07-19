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
		size?: number;
		strokeWidth?: number;
	};

	export let nodeIndex: number;
	export let layerName: string;
	export let customDensity: $$Props['customDensity'] = undefined;
	const sampleDomain: Writable<{ x: [number, number]; y: [number, number] }> =
		getContext('sampleDomain');

	const sampledOutputs: Writable<SampledOutputs<number[]>> = getContext('sampledOutputs');
	const getTfModel = getContext('getTfModel') as () => tf.Sequential;
	let tfModel = getTfModel();

	let svg: SVGSVGElement;
	let path: SVGPathElement;

	export let size = 1.5;
	$: pxSize = remToPx(size);

	export let strokeWidth = 1;

	let nodeOutputs: number[] | undefined;
	$: {
		(async () => {
			nodeOutputs = customDensity
				? await getSampledOutputForNode1D(
						tfModel,
						layerName,
						nodeIndex,
						customDensity,
						$sampleDomain.x
					)
				: $sampledOutputs && $sampledOutputs[layerName] && $sampledOutputs[layerName][nodeIndex];
		})();
	}

	$: if (nodeOutputs) {
		updateChart();
	}

	function updateChart() {
		if (!nodeOutputs) return;
		const chartWidth = pxSize;
		const chartHeight = pxSize;

		const numSamples = nodeOutputs.length;
		const xScale = d3.scaleLinear().domain($sampleDomain.x).range([0, chartWidth]);
		const yScale = d3.scaleLinear().domain($sampleDomain.y).range([chartHeight, 0]);

		const line = d3
			.line<number>()
			.x((_, i) =>
				xScale(
					d3
						.scaleLinear()
						.domain([0, numSamples - 1])
						.range($sampleDomain.x)(i)
				)
			)
			.y((d) => yScale(d));

		d3.select(path).datum(nodeOutputs).attr('d', line);
	}
</script>

<svg
	bind:this={svg}
	viewBox={`0 0 ${pxSize} ${pxSize}`}
	class="pointer-events-none"
	overflow="visible"
	{...$$restProps}
>
	<g>
		<path bind:this={path} fill="none" stroke-width={strokeWidth} class="stroke-green-500" />
	</g>
</svg>

<style>
	:global(.tick text) {
		font-size: 10px;
	}
</style>
