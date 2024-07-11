<script lang="ts">
	import type { SampledOutputs } from '$lib/structures';
	import { getContext, onMount } from 'svelte';
	import type { Writable } from 'svelte/store';

	export let nodeIndex: number;
	export let layerName: string;

	const sampledOutputs: Writable<SampledOutputs> = getContext('sampledOutputs');

	let canvas: HTMLCanvasElement;
	let ctx: CanvasRenderingContext2D;

	$: nodeOutputs = $sampledOutputs[layerName][nodeIndex];

	onMount(() => {
		ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
		if (ctx == null) throw new Error('Canvas unsupported by browser');
	});
</script>

<svg></svg>
<canvas bind:this={canvas}></canvas>
