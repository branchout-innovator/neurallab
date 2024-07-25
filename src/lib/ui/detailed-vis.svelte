<script lang="ts">
	import { getContext, onMount, onDestroy } from 'svelte';
	import type { Writable } from 'svelte/store';
	import * as tf from '@tensorflow/tfjs';
	import * as d3 from 'd3';
	import { remToPx } from '$lib/utils';
	import PredictionCurve from './prediction-curve.svelte';
	import Heatmap from './heatmap.svelte';
	import isEqual from 'lodash.isequal';
	import type { SequentialModel } from '$lib/structures';
	import { Button } from '$lib/components/ui/button';

	export let nodeIndex: number;
	export let layerName: string;

	const dataset: Writable<tf.data.Dataset<tf.TensorContainer>> = getContext('dataset');
	const model: Writable<SequentialModel> = getContext('model');
	const sampleDomain: Writable<{ x: [number, number]; y: [number, number] }> =
		getContext('sampleDomain');
	const csvColumnConfigs: Writable<{
		[key: string]: { isLabel: 'true' | 'false' };
	}> = getContext('csvColumnConfigs');

	let svg: SVGSVGElement;
	let gPoints: SVGGElement;
	let gx: SVGGElement;
	let gy: SVGGElement;
	const heatmapSize = remToPx(14);

	let inputShape: number[] | null = null;
	let randomSample: { input: Record<string, number>; output: number } | null = null;
	let activationFunction: string = '';
	let inputFeatures: string[] = [];
	let isLoading: boolean = true;
	let errorMessage: string | null = null;

	interface Sample {
		x: number;
		y: number;
		label?: number;
	}
	let testPoints: Sample[] = [];

	let zoomBehavior: d3.ZoomBehavior<SVGSVGElement, unknown>;

	const pointColorScale = d3
		.scaleLinear<string>()
		.domain([-1, 0, 1])
		.range(['#3B82F6', '#e8eaeb', '#EF4444'])
		.clamp(true);

	onMount(async () => {
		try {
			await updateVisualization();
		} catch (error) {
			console.error('Error in onMount:', error);
			errorMessage = 'An error occurred while loading the visualization.';
		} finally {
			isLoading = false;
		}
	});

	async function updateVisualization() {
		if (!$model || !$model.layers || $model.layers.length === 0) {
			throw new Error('Model is not properly loaded');
		}

		inputShape = $model.layers[0].inputShape;
		updateInputFeatures();

		if (inputShape[0] >= 3) {
			await loadRandomSample();
			getActivationFunction();
		} else {
			await loadTestPoints();
			setupZoom();
			updateChart();
		}
	}

	function updateInputFeatures() {
		inputFeatures = Object.entries($csvColumnConfigs)
			.filter(([_, config]) => config.isLabel === 'false')
			.map(([columnName, _]) => columnName);

		if (inputFeatures.length === 0) {
			inputFeatures = Array.from({ length: inputShape?.[0] || 0 }, (_, i) => `input_${i + 1}`);
		}
	}

	async function loadRandomSample() {
		if (!$dataset) throw new Error('Dataset is not loaded');

		const sampleBatch = await $dataset.take(1).toArray();
		if (sampleBatch.length === 0) throw new Error('Could not get a sample from the dataset');

		const sample = sampleBatch[0];
		const { xs, ys } = sample as { xs: number[]; ys: number[] };

		const input: Record<string, number> = {};
		for (let i = 0; i < xs.length; i++) {
			input[inputFeatures[i] || `input_${i + 1}`] = xs[i];
		}

		const inputTensor = tf.tensor2d([xs], [1, xs.length]);
		const layer = $model.layers.find((l) => l.name === layerName);
		if (!layer) throw new Error(`Layer ${layerName} not found`);

		const layerModel = tf.model({
			inputs: $model.inputs,
			outputs: layer.output
		});
		const activation = (await layerModel.predict(inputTensor)) as tf.Tensor;
		const outputValue = activation.dataSync()[nodeIndex];

		randomSample = { input, output: outputValue };

		inputTensor.dispose();
		activation.dispose();
	}

	function getActivationFunction() {
		const layer = $model.layers.find((l) => l.name === layerName);
		if (layer && layer.activation) {
			activationFunction = layer.activation.constructor.name.toLowerCase();
		} else {
			activationFunction = 'unknown';
		}
	}

	async function loadTestPoints() {
		if (!$dataset) return;
		if (isEqual(inputShape, [1])) {
			testPoints = await getNSamplesFromDataset1D($dataset, 100);
		} else if (isEqual(inputShape, [2])) {
			testPoints = await getNSamplesFromDataset2D($dataset, 100);
		}
	}

	async function getNSamplesFromDataset1D(
		dataset: tf.data.Dataset<tf.TensorContainer>,
		n: number
	): Promise<Sample[]> {
		const samples = await getNSamplesFromDataset(dataset, n);
		return samples.map((sample): Sample => {
			const { xs, ys } = sample as { xs: number[]; ys: number[] };
			const [x] = xs;
			const [y] = ys;
			return { x, y, label: 0 };
		});
	}

	async function getNSamplesFromDataset2D(
		dataset: tf.data.Dataset<tf.TensorContainer>,
		n: number
	): Promise<Sample[]> {
		const samples = await getNSamplesFromDataset(dataset, n);
		return samples.map((sample): Sample => {
			const { xs, ys } = sample as { xs: number[]; ys: number[] };
			const [x, y] = xs;
			const [label] = ys;
			return { x, y, label };
		});
	}

	async function getNSamplesFromDataset(
		dataset: tf.data.Dataset<tf.TensorContainer>,
		n: number
	): Promise<tf.TensorContainer[]> {
		const samples: tf.TensorContainer[] = [];
		const shuffledDataset = dataset.shuffle(1000);
		const samplesDataset = shuffledDataset.take(n);

		await samplesDataset.forEachAsync((sample) => {
			samples.push(sample);
		});

		return samples.slice(0, n);
	}

	function setupZoom(): void {
		zoomBehavior = d3
			.zoom<SVGSVGElement, unknown>()
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

		const xScale = d3.scaleLinear().domain(domain.x).range([0, heatmapSize]);
		const yScale = d3.scaleLinear().domain(domain.y).range([heatmapSize, 0]);

		const newXDomain = transform.rescaleX(xScale).domain();
		const newYDomain = transform.rescaleY(yScale).domain();

		$sampleDomain.x = newXDomain as [number, number];
		$sampleDomain.y = newYDomain as [number, number];

		updateChart();
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
		const domain = $sampleDomain;

		const xScale = d3.scaleLinear().domain(domain.x).range([0, heatmapSize]);
		const yScale = d3.scaleLinear().domain([domain.y[1], domain.y[0]]).range([0, heatmapSize]);

		const pointsGroup = d3.select(gPoints).data([null]);

		const points = pointsGroup
			.merge(pointsGroup)
			.selectAll<SVGCircleElement, Sample>('circle')
			.data(testPoints);

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

	$: {
		if ($csvColumnConfigs) {
			updateInputFeatures();
		}
	}
</script>

<div>
	{#if inputShape && inputShape[0] >= 3}
		<div class="rounded-lg border p-4">
			{#if randomSample}
				<h3 class="mb-2 text-lg font-semibold">Sample Activation</h3>
				<div class="mb-2">
					<h4 class="font-medium">Inputs:</h4>
					{#each Object.entries(randomSample.input) as [key, value]}
						<div>{key}: {value.toFixed(4)}</div>
					{/each}
				</div>
				<div class="mb-2">
					<h4 class="font-medium">Output:</h4>
					<div>activation: {randomSample.output.toFixed(4)}</div>
				</div>
				<div>
					<h4 class="font-medium">Activation Function:</h4>
					<div>{activationFunction}</div>
				</div>
			{:else}
				<p>No sample data available.</p>
			{/if}
		</div>
	{:else}
		<div class="relative mb-4 ml-4">
			{#if isEqual(inputShape, [1])}
				<PredictionCurve
					{nodeIndex}
					{layerName}
					class="h-56 w-56 rounded-[0.15rem]"
					customDensity={60}
					size={14}
					strokeWidth={2}
				/>
			{:else if isEqual(inputShape, [2])}
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
		<Button on:click={resetZoom} class="my-2">Reset Zoom</Button>
	{/if}
</div>
