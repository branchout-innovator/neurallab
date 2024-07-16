<script lang="ts">
	import Heatmap from './heatmap.svelte';
	import { onMount } from 'svelte';
	import * as d3 from 'd3';
	import { remToPx } from '$lib/utils';

	export let nodeIndex: number;
	export let layerName: string;

	let svg: SVGSVGElement;
	const size = 284.2;
	const margin = { top: 20, right: 20, bottom: 30, left: 40 };

	onMount(() => {
		setupAxes();
	});

	function setupAxes() {
		const heatmapSize = remToPx(14);

		const xScale = d3.scaleLinear().domain([-3, 3]).range([0, heatmapSize]);
		const yScale = d3.scaleLinear().domain([3, -3]).range([0, heatmapSize]);

		d3.select(gx)
			.call(d3.axisBottom(xScale).ticks(5).tickSize(5))
			.call((g) => g.select('.domain').remove())
			.call((g) => g.selectAll('.tick line').attr('stroke', '#888').attr('stroke-width', 0.5))
			.call((g) => g.selectAll('.tick text').attr('y', 9).attr('dy', '.71em'));
		d3.select(gy)
			.call(d3.axisLeft(yScale).ticks(5).tickSize(5))
			.call((g) => g.select('.domain').remove())
			.call((g) => g.selectAll('.tick line').attr('stroke', '#888').attr('stroke-width', 0.5))
			.call((g) => g.selectAll('.tick text').attr('x', -9).attr('dy', '.32em'));
	}

	let gx: SVGGElement;
	let gy: SVGGElement;
</script>

<div>
	<div class="relative mb-4 ml-4">
		<Heatmap {nodeIndex} {layerName} customDensity={60} class=" h-56 w-56 rounded" />
		<svg
			bind:this={svg}
			width={size}
			height={size}
			class="pointer-events-none absolute left-0 top-0"
			overflow="visible"
		>
			<g bind:this={gx} class="translate-y-56"></g>
			<g bind:this={gy}></g>
		</svg>
	</div>
</div>
