<script lang="ts">
	import Heatmap from './heatmap.svelte';
	import { onMount } from 'svelte';
	import * as d3 from 'd3';
	import { remToPx } from '$lib/utils';
	import * as tf from '@tensorflow/tfjs';
	//import type { Dataset } from '@tensorflow/tfjs';

	export let nodeIndex: number;
	export let layerName: string;
	export let dataset: tf.data.Dataset<tf.TensorContainer>;

	let svg: SVGSVGElement;
	const size = 284.2;

	let testPoints: { x: number; y: number; label: number }[] = [];

	const pointColorScale = d3
		.scaleLinear<string>()
		.domain([-1, 0, 1])
		.range(['#3B82F6', '#e8eaeb', '#EF4444'])
		.clamp(true);

	onMount(async () => {
		await loadTestPoints();
		setupAxes();
		drawTestPoints();
	});

	async function loadTestPoints() {
		/*
		// Load tf dataset here
		const dataset = tf.data.array([
			{ xs: [1, 2], ys: 1 },
			{ xs: [-1, -2], ys: -1 },
			{ xs: [-1.1, -2.1], ys: 1 }
		]);*/

		await dataset.forEachAsync((element) => {
			const x = element.xs[0];
			const y = element.xs[1];
			const label = element.ys;
			testPoints.push({ x, y, label });
		});

		testPoints = testPoints;
	}

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

	function drawTestPoints() {
		const heatmapSize = remToPx(14);
		const xScale = d3.scaleLinear().domain([-3, 3]).range([0, heatmapSize]);
		const yScale = d3.scaleLinear().domain([3, -3]).range([0, heatmapSize]);

		const pointsGroup = d3.select(svg).append('g').attr('class', 'points');

		pointsGroup
			.selectAll('circle')
			.data(testPoints)
			.enter()
			.append('circle')
			.attr('cx', (d) => xScale(d.x))
			.attr('cy', (d) => yScale(d.y))
			.attr('r', 3)
			.style('fill', (d) => pointColorScale(d.label))
			.style('stroke', 'white')
			.style('stroke-width', '0.5');
	}

	let gx: SVGGElement;
	let gy: SVGGElement;
</script>

<div>
	<div class="relative mb-4 ml-4">
		<Heatmap {nodeIndex} {layerName} customDensity={60} class="h-56 w-56 rounded" />
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
