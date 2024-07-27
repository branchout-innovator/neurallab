<script lang="ts">
	import { getSampledOutputForNode, type SampledOutputs } from '$lib/structures';
	import { getContext, onMount } from 'svelte';
	import type { Writable } from 'svelte/store';
	import * as d3 from 'd3';
	import type { HTMLAttributes } from 'svelte/elements';
	import * as tf from '@tensorflow/tfjs';
	type $$Props = HTMLAttributes<HTMLCanvasElement> & {
		nodeIndex: number;
		layerName: string;
		customDensity?: number;
	};

	export let nodeIndex: number;
	export let layerName: string;
	export let customDensity: $$Props['customDensity'] = undefined;

	const sampleDomain: Writable<{ x: [number, number]; y: [number, number] }> =
		getContext('sampleDomain');

	const sampledOutputs: Writable<SampledOutputs<number[][]>> = getContext('sampledOutputs');
	const getTfModel = getContext('getTfModel') as () => tf.Sequential;
	let tfModel = getTfModel();

	let canvas: HTMLCanvasElement;
	let svg: SVGSVGElement;
	let ctx: CanvasRenderingContext2D;

	let chartWidth: number | undefined;
	let chartHeight: number | undefined;
	let nodeOutputs: number[][] | undefined;
	$: {
		(async () => {
			nodeOutputs = customDensity
				? await getSampledOutputForNode(
						tfModel,
						layerName,
						nodeIndex,
						customDensity,
						$sampleDomain.x,
						$sampleDomain.y
					)
				: $sampledOutputs &&
					$sampledOutputs[layerName] &&
					$sampledOutputs[layerName].values[nodeIndex];
		})();
	}

	$: {
		if (nodeOutputs) {
			chartWidth = customDensity || nodeOutputs.length;
			chartHeight = customDensity || nodeOutputs.length;
		}
		updateHeatmap();
	}

	onMount(() => {
		if (ctx == null) ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
		if (ctx == null) {
			console.error('Canvas unsupported by browser');
			throw new Error('Canvas unsupported by browser');
		}

		setupAxes();
		if (nodeOutputs) {
			updateHeatmap();
		}
	});

	function updateHeatmap() {
		if (!ctx || !nodeOutputs || !chartWidth || !chartHeight) {
			return;
		}
		canvas.width = chartWidth;
		canvas.height = chartHeight;

		const numSamples = customDensity || nodeOutputs.length;

		const colorScale = d3
			.scaleSequential(d3.interpolateRdBu)
			.domain([
				d3.max(nodeOutputs, (row) => d3.max(row)) || 1,
				d3.min(nodeOutputs, (row) => d3.min(row)) || -1
			]);

		const imageData = ctx.createImageData(numSamples, numSamples);
		for (let y = 0; y < numSamples; y++) {
			for (let x = 0; x < numSamples; x++) {
				const value = nodeOutputs[y][x];
				const color = d3.rgb(colorScale(value));
				const index = (y * numSamples + x) * 4;
				imageData.data[index] = color.r;
				imageData.data[index + 1] = color.g;
				imageData.data[index + 2] = color.b;
				imageData.data[index + 3] = 255; // Alpha
			}
		}
		ctx.putImageData(imageData, 0, 0);
	}

	function setupAxes() {
		/*
        // Create scales for axes
        const xScale = d3.scaleLinear()
            .domain(xDomain)
            .range([0, chartWidth]);

        const yScale = d3.scaleLinear()
            .domain(yDomain)
            .range([chartHeight, 0]);

        // Create SVG for axes
        const svgSelection = d3.select(svg)
            .attr('width', width)
            .attr('height', height);

        const g = svgSelection.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Add X axis
        g.append('g')
            .attr('transform', `translate(0,${chartHeight})`)
            .call(d3.axisBottom(xScale))
            .attr('class', 'x-axis');

        // Add Y axis
        g.append('g')
            .call(d3.axisLeft(yScale))
            .attr('class', 'y-axis');

        // Add X axis label
        g.append('text')
            .attr('transform', `translate(${chartWidth/2},${chartHeight + margin.bottom - 10})`)
            .style('text-anchor', 'middle')
            .text('X Axis')
            .attr('class', 'axis-label');

        // Add Y axis label
        g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', 0 - margin.left + 20)
            .attr('x', 0 - (chartHeight / 2))
            .attr('dy', '1em')
            .style('text-anchor', 'middle')
            .text('Y Axis')
            .attr('class', 'axis-label');
            */
	}
</script>

<canvas bind:this={canvas} {...$$restProps}></canvas>
