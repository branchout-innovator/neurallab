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
        ["#888","#888"]);
        updatePoints(prevPoints);
    });

    async function updatePoints(points: number[]) {
        points.forEach(point => {
            updateGraph(point);
        });
    }

	const sampledOutputs: Writable<SampledOutputs<number[][]>> = getContext('sampledOutputs');
	const getTfModel = getContext('getTfModel') as () => tf.Sequential;


	export function updateGraph(loss: number) {
		lineChart?.addDataPoint(loss);
        setupAxes();
	}

	function setupAxes() {
		const xScale = d3.scaleLinear().domain([0, prevPoints.length]).range([0, 320]);
		const yScale = d3.scaleLinear().domain([lineChart.minY, lineChart.maxY]).range([230, 0]);

		d3.select(gx)
            .call(d3.axisBottom(xScale).ticks(5).tickSize(2))
			.call((g) => g.select('.domain').remove())
			.call((g) => g.selectAll('.tick line').attr('stroke', '#888').attr('stroke-width', 0.5))
			.call((g) => g.selectAll('.tick text').attr('y', 6).attr('dy', '.71em'));
		d3.select(gy)
			.call(d3.axisLeft(yScale).ticks(5).tickSize(2))
			.call((g) => g.select('.domain').remove())
			.call((g) => g.selectAll('.tick line').attr('stroke', '#888').attr('stroke-width', 0.5))
			.call((g) => g.selectAll('.tick text').attr('x', -6).attr('dy', '.32em'));
	}
</script>
<div class="relative mb-4 ml-4">
    <div id="linechart" {...$$restProps}></div>
    <svg class="absolute inset-0 h-full w-full" overflow="visible">
        <g bind:this={gx} class="translate-y-60"></g>
        <g bind:this={gy}></g>
    </svg>
</div>