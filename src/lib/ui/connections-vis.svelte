<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import { browser } from '$app/environment';
	import { Button } from '$lib/components/ui/button/index.js';
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';
	import { remToPx } from '$lib/utils';

	export let leftLayerHeights: number[];
	export let rightLayerHeights: number[];
	export let canvasWidth: number;
	export let weights: tf.Tensor;

	let svgElement: SVGSVGElement;
	let paths: {
		d: string;
		color: string;
		strokeWidth: number;
		weight: number;
		normalizedWeight: number;
	}[] = [];

	function getMaxAbsWeight(
		arr:
			| number
			| number[]
			| number[][]
			| number[][][]
			| number[][][][]
			| number[][][][][]
			| number[][][][][][]
	): number {
		if (typeof arr === 'number') {
			return Math.abs(arr);
		}
		return Math.max(...arr.map(getMaxAbsWeight));
	}

	const updateSvg = async (weights: tf.Tensor) => {
		if (!browser) return;
		const weightsArray = (await weights.array()) as number[][];
		if (typeof weightsArray === 'number') return;
		const maxWeight = getMaxAbsWeight(weightsArray);
		const endHeight = remToPx(1.1);

		paths = [];
		// if (leftLayerHeights.length * rightLayerHeights.length > 500) return;

		for (let i = 0; i < leftLayerHeights.length; i++) {
			for (let j = 0; j < rightLayerHeights.length; j++) {
				const weight = weightsArray[i][j];
				const normalizedWeight = weight / maxWeight;

				const rightConnectionSpacing = endHeight / leftLayerHeights.length;
				const startX = 0;
				const startY = leftLayerHeights[i];
				const endX = canvasWidth;
				const endY =
					rightLayerHeights[j] + (i - (leftLayerHeights.length - 1) / 2) * rightConnectionSpacing;

				const controlPoint1X = startX + canvasWidth / 3;
				const controlPoint1Y = startY;
				const controlPoint2X = endX - canvasWidth / 3;
				const controlPoint2Y = endY;

				const d = `M ${startX} ${startY} C ${controlPoint1X} ${controlPoint1Y}, ${controlPoint2X} ${controlPoint2Y}, ${endX} ${endY}`;

				const color = getColor(normalizedWeight);
				const strokeWidth = Math.abs(normalizedWeight) * 2 + 0.5;

				paths.push({ d, color, strokeWidth, weight, normalizedWeight });
			}
		}
	};

	$: {
		updateSvg(weights);
	}

	function getColor(normalizedWeight: number): string {
		return normalizedWeight > 0 ? '#EF4444' : '#3B82F6';
	}

	$: canvasHeight = Math.max(...leftLayerHeights, ...rightLayerHeights) + 20;
</script>

<svg bind:this={svgElement} width={canvasWidth} height={canvasHeight} class="mt-[4.25rem]">
	{#each paths as path}
		<path
			d={path.d}
			stroke={path.color}
			fill="none"
			stroke-width={path.strokeWidth}
			opacity={Math.abs(path.normalizedWeight)}
		/>
	{/each}

	<!-- {#each leftLayerHeights as height, i}
		<circle cx="0" cy={height} r="5" fill="#333" />
	{/each}

	{#each rightLayerHeights as height, i}
		<circle cx={canvasWidth - 2.5} cy={height} r="5" fill="#333" />
	{/each} -->
</svg>
