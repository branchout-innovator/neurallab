<script lang="ts">
	import { Button } from '$lib/components/ui/button/index.js';
	import * as Popover from '$lib/components/ui/popover/index.js';
	import logo from '$lib/images/image0.png';
	import { getSampledOutputForNode, type SampledOutputs } from '$lib/structures';
	import { getContext, onMount } from 'svelte';
	import type { Writable } from 'svelte/store';
	import * as d3 from 'd3';
	import type { HTMLAttributes } from 'svelte/elements';
	import * as tf from '@tensorflow/tfjs';
	type $$Props = HTMLAttributes<HTMLCanvasElement> & {
		/*nodeIndex: number;
		layerName: string;*/
		rgb: boolean;
		nodeIndex: number;
		layerName: string;
	};

	/*export let nodeIndex: number;
	export let layerName: string;*/
	export let nodeIndex: number;
	export let layerName: string;
	export let rgb: boolean;
	let image: number[][] = [[]];	

	const sampleDomain: Writable<{ x: [number, number]; y: [number, number] }> =
		getContext('sampleDomain');

	const sampledOutputs: Writable<SampledOutputs<number[][]>> = getContext('sampledOutputs');
	const getTfModel = getContext('getTfModel') as () => tf.Sequential;
	let tfModel = getTfModel();

	$: {
		let sampled = $sampledOutputs &&
		$sampledOutputs[layerName];
		if (sampled) {
			image = selectImages(sampled.values)[nodeIndex];
		}
				
	}

	let canvas: HTMLCanvasElement;
	let svg: SVGSVGElement;
	let ctx: CanvasRenderingContext2D;

	let chartWidth: number | undefined;
	let chartHeight: number | undefined;
	let nodeOutputs: number[][] | undefined;
	$: {
		/*(async () => {
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
		})();*/
		nodeOutputs = image;
	}

	$: {
		if (nodeOutputs) {
			chartWidth = nodeOutputs[0].length;
			chartHeight = nodeOutputs.length;
		}
		updateHeatmap();
	}

	onMount(() => {
		console.log(image);
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

	function selectImages(output: number[][][]): number[][][] {
		let images: number[][][] = [];
		for (let i = 0; i < output[0][0].length; i++) {
			let image: number[][] = [];
			for (let j = 0; j < output.length; j++) {
				let row: number[] = [];
				for (let k = 0; k < output[0].length; k++) {
					row.push(output[j][k][i]);
				}
				image.push(row);
			}
			images.push(image);
		}
		return images;
	}


	function updateHeatmap() {
		if (!ctx || !nodeOutputs || !chartWidth || !chartHeight) {
			return;
		}
		canvas.width = chartWidth;
		canvas.height = chartHeight;

		const numSamples = nodeOutputs.length;
		const scale = getScale(nodeOutputs);
		const imageData = ctx.createImageData(numSamples, numSamples);
		for (let y = 0; y < numSamples; y++) {
			for (let x = 0; x < numSamples; x++) {
				const value = nodeOutputs[y][x];
				const index = (y * numSamples + x) * 4;
				if (rgb) {
					if (value < 0) {
						imageData.data[index] = 255;
						imageData.data[index + 1] = 255-(Math.abs(value))/(Math.abs(scale[0]))*255;
						imageData.data[index + 2] = 255-(Math.abs(value))/(Math.abs(scale[0]))*255;;
						imageData.data[index + 3] = 255; // Alpha
					}
					else {
						imageData.data[index] = 255-(Math.abs(value))/(Math.abs(scale[1]))*255;
						imageData.data[index + 1] = 255-(Math.abs(value))/(Math.abs(scale[1]))*255;
						imageData.data[index + 2] = 255;
						imageData.data[index + 3] = 255; // Alpha
					}
				}
				else {
					imageData.data[index] = value;
					imageData.data[index + 1] = value;
					imageData.data[index + 2] = value;
					imageData.data[index + 3] = 255;
				}
			}
		}
		ctx.putImageData(imageData, 0, 0);
	}

	function getScale(image: number[][]) {
		let maxim = Math.max(...image.map((x) => {return Math.max(...x)}))
		let minim = Math.min(...image.map((x) => {return Math.min(...x)}))
		return [minim, maxim];
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

