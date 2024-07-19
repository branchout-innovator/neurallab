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
	let gPoints: SVGGElement;
	const heatmapSize = remToPx(14);

	interface Sample {
		x: number;
		y: number;
		label?: number;
	}
	let testPoints: Sample[] = [];

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
		setupZoom();
		updateChart();
	});

	class EnoughSamplesCollectedError extends Error {
		constructor() {
			super('EnoughSamplesCollected');
			this.name = 'EnoughSamplesCollectedError';
		}
	}

	async function getNSamplesFromDataset(
		dataset: tf.data.Dataset<tf.TensorContainer>,
		n: number
	): Promise<tf.TensorContainer[]> {
		const samples: tf.TensorContainer[] = [];

		// Shuffle the dataset to get random samples
		const shuffledDataset = dataset.shuffle(1000);

		// Take n samples
		const samplesDataset = shuffledDataset.take(n);

		// Collect the samples
		try {
			await samplesDataset.forEachAsync((sample) => {
				samples.push(sample);
				if (samples.length >= n) {
					// Stop iterating once we have n samples
					throw new EnoughSamplesCollectedError();
				}
			});
		} catch (error: unknown) {
			if (error instanceof EnoughSamplesCollectedError) {
				// We've collected enough samples, so we can ignore this error
			} else if (error instanceof Error) {
				// If it's another kind of Error, rethrow it
				throw error;
			} else {
				// If it's not an Error object at all, throw a new Error
				throw new Error('An unknown error occurred');
			}
		}

		return samples.slice(0, n); // Ensure we return exactly n samples
	}

	async function getNSamplesFromDataset2D(
		dataset: tf.data.Dataset<tf.TensorContainer>,
		n: number
	): Promise<Sample[]> {
		return (await getNSamplesFromDataset(dataset, n)).map((sample): Sample => {
			const { xs, ys } = sample as { xs: number[]; ys: number[] };
			const [x, y] = xs;
			const [label] = ys;
			return { x, y, label };
		});
	}

	async function getNSamplesFromDataset1D(
		dataset: tf.data.Dataset<tf.TensorContainer>,
		n: number
	): Promise<Sample[]> {
		return (await getNSamplesFromDataset(dataset, n)).map((sample): Sample => {
			const { xs, ys } = sample as { xs: number[]; ys: number[] };
			const [x] = xs;
			const [y] = ys;
			return { x, y, label: 0 };
		});
	}

	async function loadTestPoints() {
		// Load tf dataset here
		if (!$dataset) return;
		const inputShape = $model.layers[0].inputShape;
		if (isEqual(inputShape, [1])) {
			testPoints = await getNSamplesFromDataset1D($dataset, 100);
		} else if (isEqual(inputShape, [2])) {
			testPoints = await getNSamplesFromDataset2D($dataset, 100);
		}
	}

	function setupAxes() {
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
		const domain = $sampleDomain;

		const xScale = d3.scaleLinear().domain(domain.x).range([0, heatmapSize]);
		const yScale = d3.scaleLinear().domain([domain.y[1], domain.y[0]]).range([0, heatmapSize]);

		const pointsGroup = d3.select(gPoints).data([null]);

		const points = pointsGroup
			.merge(pointsGroup)
			.selectAll<SVGCircleElement, { x: number; y: number; label: number }>('circle')
			.data(testPoints);
		console.log('drawing ', testPoints);

		points
			.enter()
			.append('circle')
			.merge(points)
			.attr('cx', (d) => xScale(d.x))
			.attr('cy', (d) => yScale(d.y))
			.attr('r', 3)
			.style('fill', (d) => pointColorScale(d.label ?? 1))
			.style('fill-opacity', 0.7)
			.style('stroke', 'white')
			.style('stroke-width', '0.5');

		points.exit().remove();
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

		$sampleDomain.x = newXDomain as [number, number];
		$sampleDomain.y = newYDomain as [number, number];

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
		if (gPoints) {
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
			/>
		{:else if isEqual($model.layers[0].inputShape, [2])}
			<Heatmap {nodeIndex} {layerName} class="h-56 w-56 rounded-[0.15rem]" customDensity={60} />
		{/if}
		<svg class="absolute inset-0 h-full w-full" overflow="visible">
			<g bind:this={gx} class="translate-y-56"></g>
			<g bind:this={gy}></g>
		</svg>
		<svg bind:this={svg} class="absolute inset-0 h-full w-full" overflow="hidden">
			<g bind:this={gPoints} overflow="hidden"></g>
		</svg>
	</div>
	<Button on:click={resetZoom}>Reset Zoom</Button>
</div>
