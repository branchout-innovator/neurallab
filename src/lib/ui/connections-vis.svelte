<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import { browser } from '$app/environment';

	export let leftLayerHeights: number[];
	export let rightLayerHeights: number[];
	export let canvasWidth: number;
	export let weights: tf.Tensor;

	let svgElement: SVGSVGElement;
	let paths: { d: string; color: string; strokeWidth: number }[] = [];

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
		const weightsArray = await weights.array();
		if (typeof weightsArray === 'number') return;
		const maxWeight = getMaxAbsWeight(weightsArray);

		paths = [];

		for (let i = 0; i < leftLayerHeights.length; i++) {
			for (let j = 0; j < rightLayerHeights.length; j++) {
				const weight = (weightsArray[i] as number[])[j];
				const normalizedWeight = weight / maxWeight;

				const startX = 0;
				const startY = leftLayerHeights[i];
				const endX = canvasWidth;
				const endY = rightLayerHeights[j];

				const controlPoint1X = startX + canvasWidth / 3;
				const controlPoint1Y = startY;
				const controlPoint2X = endX - canvasWidth / 3;
				const controlPoint2Y = endY;

				const d = `M ${startX} ${startY} C ${controlPoint1X} ${controlPoint1Y}, ${controlPoint2X} ${controlPoint2Y}, ${endX} ${endY}`;

				const color = getColor(normalizedWeight);
				const strokeWidth = Math.abs(normalizedWeight) * 3 + 0.5;

				paths.push({ d, color, strokeWidth });
			}
		}
	};

	$: {
		updateSvg(weights);
	}

	function getColor(normalizedWeight: number): string {
		const r = normalizedWeight > 0 ? Math.round(normalizedWeight * 255) : 0;
		const b = normalizedWeight < 0 ? Math.round(-normalizedWeight * 255) : 0;
		return `rgb(${r}, 0, ${b})`;
	}

	$: canvasHeight = Math.max(...leftLayerHeights, ...rightLayerHeights) + 20;
</script>

<svg bind:this={svgElement} width={canvasWidth} height={canvasHeight} class="mt-[4.25rem]">
	{#each paths as path}
		<path d={path.d} stroke={path.color} fill="none" stroke-width={path.strokeWidth} />
	{/each}

	<!-- {#each leftLayerHeights as height, i}
		<circle cx="0" cy={height} r="5" fill="#333" />
	{/each}

	{#each rightLayerHeights as height, i}
		<circle cx={canvasWidth - 2.5} cy={height} r="5" fill="#333" />
	{/each} -->
</svg>
