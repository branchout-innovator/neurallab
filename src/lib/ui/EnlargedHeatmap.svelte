<script lang="ts">
	import Heatmap from './heatmap.svelte';
	import { onMount } from 'svelte';
	import * as d3 from 'd3';

	export let nodeIndex: number;
	export let layerName: string;
	//export let sample_x_domain: [number, number]
	//export let sample_y_domain: [number, number]

	let svg: SVGSVGElement;
	const size = 284.2;
	const margin = { top: 20, right: 20, bottom: 30, left: 40 };
	const chartSize = size - margin.left - margin.right;

	onMount(() => {
		setupAxes();
	});

	function setupAxes() {
		const svgSelection = d3.select(svg);
		svgSelection.selectAll('*').remove();

		const xScale = d3.scaleLinear().domain([0, 1]).range([0, chartSize]);
		const yScale = d3.scaleLinear().domain([1, 0]).range([0, chartSize]);

		const g = svgSelection.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

		g.append('g')
			.attr('transform', `translate(0,${size})`)
			.call(d3.axisBottom(xScale).ticks(5).tickSize(5))
			.call((g) => g.select('.domain').remove()) // Remove the axis line
			.call((g) => g.selectAll('.tick line').attr('stroke', '#888').attr('stroke-width', 0.5)) // Style tick lines
			.call((g) => g.selectAll('.tick text').attr('y', 9).attr('dy', '.71em')); // Adjust text position

		// Add Y axis
		g.append('g')
			.call(d3.axisLeft(yScale).ticks(5).tickSize(5))
			.call((g) => g.select('.domain').remove()) // Remove the axis line
			.call((g) => g.selectAll('.tick line').attr('stroke', '#888').attr('stroke-width', 0.5)) // Style tick lines
			.call((g) => g.selectAll('.tick text').attr('x', -9).attr('dy', '.32em')); // Adjust text position
	}
</script>

<div class="relative" style="width:{size}px; height:{size}px;">
	<Heatmap
		{nodeIndex}
		{layerName}
		customDensity={60}
		class="absolute left-10 top-5 h-56 w-56 rounded"
	/>
	<svg bind:this={svg} width={size} height={size} class="pointer-events-none absolute left-0 top-0"
	></svg>
</div>

<style>
	:global(.axis-label) {
		font-size: 12px;
		fill: #333;
	}
</style>
