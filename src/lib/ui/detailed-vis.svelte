<script lang="ts">
	import Heatmap from './heatmap.svelte';
	import { getContext, onDestroy, onMount } from 'svelte';
	import * as d3 from 'd3';
	import { remToPx } from '$lib/utils';
	import * as tf from '@tensorflow/tfjs';
	import PredictionCurve from './prediction-curve.svelte';
	import isEqual from 'lodash.isequal';
	import type { SequentialModel } from '$lib/structures';
	import type { Writable } from 'svelte/store';
	import { zoom } from 'd3';
	import { Button } from '$lib/components/ui/button';
	//import type { Dataset } from '@tensorflow/tfjs';

	export let nodeIndex: number;
	export let layerName: string;
	const dataset: Writable<tf.data.Dataset<tf.TensorContainer>> = getContext('dataset');

	const model: Writable<SequentialModel> = getContext('model');

	let svg: SVGSVGElement;

	let testPoints: { x: number; y: number; label: number }[] = [];

	const sampleDomain: Writable<{ x: [number, number]; y: [number, number] }> =
		getContext('sampleDomain');

	let zoomBehavior: d3.ZoomBehavior<SVGSVGElement, unknown>;

	const pointColorScale = d3
		.scaleLinear<string>()
		.domain([-1, 0, 1])
		.range(['#3B82F6', '#e8eaeb', '#EF4444'])
		.clamp(true);

	onMount(async () => {
		await loadTestPoints();
		setupAxes();
		drawTestPoints();
		setupZoom();
	});

	async function loadTestPoints() {
		// Load tf dataset here
		/*
		const dataset = tf.data.array([
			{ xs: [1, 2], ys: 1 },
			{ xs: [-1, -2], ys: -1 },
			{ xs: [-1.1, -2.1], ys: 1 }
		]);*/
		if (!dataset) return;
		testPoints = [];
		let count = 0;
		await $dataset.forEachAsync((element) => {
			const data = element as { xs: tf.Tensor; ys: tf.Tensor };
			const x = data.xs.dataSync()[0];
			const y = data.xs.dataSync()[1];
			const label = data.ys.dataSync()[0];
			testPoints.push({ x, y, label });
			console.log({ x, y, label });
			count++;
		});
		console.log('test points: ' + count);

		testPoints = testPoints;
	}

	function setupAxes() {
		const heatmapSize = remToPx(14);

		const xScale = d3.scaleLinear().domain($sampleDomain.x).range([0, heatmapSize]);
		const yScale = d3.scaleLinear().domain($sampleDomain.y).range([heatmapSize, 0]);

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

	function setupZoom(): void {
		const heatmapSize = remToPx(14);

		zoomBehavior = zoom<SVGSVGElement, unknown>()
			.scaleExtent([0.5, 5])
			.extent([
				[0, 0],
				[heatmapSize, heatmapSize]
			])
			.on('zoom', zoomed);

		d3.select(svg).call(zoomBehavior);
	}

	function zoomed(event: d3.D3ZoomEvent<SVGSVGElement, unknown>): void {
		const { transform } = event;
		const domain = $sampleDomain;
		const heatmapSize = remToPx(14);

		const xScale = d3.scaleLinear().domain(domain.x).range([0, heatmapSize]);
		const yScale = d3.scaleLinear().domain(domain.y).range([heatmapSize, 0]);

		const newXDomain = transform.rescaleX(xScale).domain();
		const newYDomain = transform.rescaleY(yScale).domain();

		sampleDomain.update((d) => ({
			x: newXDomain as [number, number],
			y: newYDomain as [number, number]
		}));

		updateChart();
	}

	function updateChart(): void {
		setupAxes();
		drawTestPoints();
	}

	function resetZoom(): void {
		sampleDomain.set({
			x: [-3, 3],
			y: [-3, 3]
		});
		d3.select(svg).call(zoomBehavior.transform, d3.zoomIdentity);
		updateChart();
	}

	const unsubscribe = sampleDomain.subscribe(() => {
		if (svg) {
			updateChart();
		}
	});

	onDestroy(unsubscribe);
</script>

<div>
	<div class="relative mb-4 ml-4">
		{#if isEqual($model.layers[0].inputShape, [1])}
			<PredictionCurve
				{nodeIndex}
				{layerName}
				class="h-56 w-56 rounded-[0.15rem]"
				customDensity={60}
				size={14}
				strokeWidth={2}
				xDomain={$sampleDomain.x}
				yDomain={$sampleDomain.y}
			/>
		{:else if isEqual($model.layers[0].inputShape, [2])}
			<Heatmap
				{nodeIndex}
				{layerName}
				class="h-56 w-56 rounded-[0.15rem]"
				customDensity={60}
				xDomain={$sampleDomain.x}
				yDomain={$sampleDomain.y}
			/>
		{/if}
		<svg bind:this={svg} class="absolute left-0 top-0" overflow="visible">
			<g bind:this={gx} class="translate-y-56"></g>
			<g bind:this={gy}></g>
		</svg>
	</div>
	<Button on:click={resetZoom}>Reset Zoom</Button>
</div>
