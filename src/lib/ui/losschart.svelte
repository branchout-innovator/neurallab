<script lang="ts">
	import { getSampledOutputForNode, type SampledOutputs } from '$lib/structures';
	import { getContext, onMount } from 'svelte';
	import type { Writable } from 'svelte/store';
	import * as d3 from 'd3';
	import type { HTMLAttributes } from 'svelte/elements';
	import * as tf from '@tensorflow/tfjs';
    import {AppendingLineChart} from './linechart';
	type $$Props = HTMLAttributes<HTMLCanvasElement> 

    let lineChart = new AppendingLineChart(d3.select("#linechart"),
        ["#777", "black"]);
	const sampledOutputs: Writable<SampledOutputs<number[][]>> = getContext('sampledOutputs');
	const getTfModel = getContext('getTfModel') as () => tf.Sequential;
	let tfModel = getTfModel();


	onMount(() => {
		updateHeatmap();
	});

	function updateHeatmap() {
		lineChart.addDataPoint(1);
        lineChart.addDataPoint(1);
        lineChart.addDataPoint(1);
        lineChart.addDataPoint(1);
        lineChart.addDataPoint(1);
        lineChart.addDataPoint(1);
        lineChart.addDataPoint(1);
        lineChart.addDataPoint(1);
        lineChart.addDataPoint(1);
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

<div id="linechart" {...$$restProps}></div>
