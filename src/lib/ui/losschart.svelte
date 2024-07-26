<script lang="ts">
	import { getSampledOutputForNode, type SampledOutputs } from '$lib/structures';
	import { getContext, onMount } from 'svelte';
	import type { Writable } from 'svelte/store';
	import * as d3 from 'd3';
	import type { HTMLAttributes } from 'svelte/elements';
	import * as tf from '@tensorflow/tfjs';
    import {AppendingLineChart} from './linechart';

	type $$Props = HTMLAttributes<HTMLCanvasElement> & {
        prevPoints: number[];
    }

    export let prevPoints: number[];

    let lineChart: AppendingLineChart;
    onMount(() => {
    lineChart = new AppendingLineChart(d3.select("#linechart"),
        ["#777", "black"]);
        prevPoints.forEach(point => {
            updateGraph(point);
        });
    });
	const sampledOutputs: Writable<SampledOutputs<number[][]>> = getContext('sampledOutputs');
	const getTfModel = getContext('getTfModel') as () => tf.Sequential;


	export function updateGraph(loss: number) {
		lineChart?.addDataPoint(loss);
	}
</script>

<div id="linechart" {...$$restProps}></div>
