<script lang="ts">
	import { getSampledOutputForNode, type SampledOutputs } from '$lib/structures';
	import { getContext, onMount } from 'svelte';
	import type { Writable } from 'svelte/store';
	import * as d3 from 'd3';
	import type { HTMLAttributes } from 'svelte/elements';
	import * as tf from '@tensorflow/tfjs';
    import {AppendingLineChart} from '../../lib/ui/linechart';

	type $$Props = HTMLAttributes<HTMLCanvasElement> & {
        prevPoints: number[];
    }

    export let prevPoints: number[];
    let lineChart: AppendingLineChart;

    let gx: SVGGElement;
	let gy: SVGGElement;

    onMount(() => {
    lineChart = new AppendingLineChart(d3.select("#linechart"),
        ["gray","gray"]);
        prevPoints.forEach(point => {
            updateGraph(point);
        });
    });
	const sampledOutputs: Writable<SampledOutputs<number[][]>> = getContext('sampledOutputs');
	const getTfModel = getContext('getTfModel') as () => tf.Sequential;


	export function updateGraph(loss: number) {
		lineChart?.addDataPoint(loss);
        setupAxes();
	}
</script>
<div class="relative mb-4 ml-4">
    <div id="linechart" {...$$restProps}></div>
    <svg class="absolute inset-0 h-full w-full" overflow="visible">
        <g bind:this={gx} class="translate-y-60"></g>
        <g bind:this={gy}></g>
    </svg>
</div>