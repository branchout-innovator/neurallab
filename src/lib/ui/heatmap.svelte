<!--<script lang="ts">
    import type { SampledOutputs } from '$lib/structures';
    import { getContext, onMount } from 'svelte';
    import type { Writable } from 'svelte/store';
    import * as d3 from 'd3';

    export let nodeIndex: number;
    export let layerName: string;
    export let width = 700;
    export let height = 700;
    export let xDomain: [number, number] = [-6, 6];
    export let yDomain: [number, number] = [-6, 6];

    const sampledOutputs: Writable<SampledOutputs> = getContext('sampledOutputs');

    let canvas: HTMLCanvasElement;
    let svg: SVGSVGElement;
    let ctx: CanvasRenderingContext2D;

    const margin = { top: 40, right: 40, bottom: 60, left: 80 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    $: nodeOutputs = $sampledOutputs && $sampledOutputs[layerName] && $sampledOutputs[layerName][nodeIndex];

    $: {
        console.log('sampledOutputs:', $sampledOutputs);
        console.log('layerName:', layerName);
        console.log('nodeIndex:', nodeIndex);
        console.log('nodeOutputs:', nodeOutputs);
    }

    onMount(() => {
        console.log('Component mounted');
        ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
        if (ctx == null) {
            console.error('Canvas unsupported by browser');
            throw new Error('Canvas unsupported by browser');
        }
        console.log('Canvas context:', ctx);

        setupAxes();
        if (nodeOutputs) {
            updateHeatmap();
        }
    });

    $: if (nodeOutputs) {
        console.log('nodeOutputs changed, updating heatmap');
        updateHeatmap();
    }

    function updateHeatmap() {
        console.log('Updating heatmap');
        if (!ctx || !nodeOutputs) {
            console.log('ctx or nodeOutputs is null', { ctx, nodeOutputs });
            return;
        }

        const numSamples = nodeOutputs.length;

        // Create color scale
        const colorScale = d3.scaleSequential(d3.interpolateRdBu)
            .domain([d3.min(nodeOutputs, row => d3.min(row)) || -1, 
                     d3.max(nodeOutputs, row => d3.max(row)) || 1]);

        // Draw heatmap on canvas
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
       
    }
</script>

<div class="heatmap-container" style="position: relative; width: {width}px; height: {height}px;">
    <canvas 
        bind:this={canvas} 
        width={chartWidth} 
        height={chartHeight}
        style="position: absolute; left: {margin.left}px; top: {margin.top}px;"
    ></canvas>
    <svg 
        bind:this={svg} 
        style="position: absolute; left: 0; top: 0;"
    ></svg>
</div>

<style>
    .heatmap-container {
        display: inline-block;
    }
</style>-->