<script lang="ts">
    import { onMount } from 'svelte';
    import * as d3 from 'd3';
    import * as tf from '@tensorflow/tfjs';
    import Button from '$lib/components/ui/button/button.svelte';
	import type { Conv2DLayer, DenseLayer, MaxPoolingLayer, SequentialModel } from '$lib/structures';
	import { remToPx } from '$lib/utils';
	import Minus from 'lucide-svelte/icons/minus';
	import Plus from 'lucide-svelte/icons/plus';
	import { getContext, setContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { browser } from '$app/environment';
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';
	import Heatmap from './heatmap.svelte';
	import * as HoverCard from '$lib/components/ui/hover-card';
	import EnlargedHeatmap from './detailed-vis.svelte';
	import PredictionCurve from './prediction-curve.svelte';
	import isEqual from 'lodash.isequal';
	import ActivationColor from './activation-color.svelte';
	import Losschart from './losschart.svelte';
    import Bone from './bone.svelte';


    export let model: tf.LayersModel;
    export let layerName: string;
    // export let inputImage: tf.Tensor3D;
    export let layer: MaxPoolingLayer;

    let svg: SVGSVGElement;
    let width = 600;
    let height = 400;
    let padding = 40;

    onMount(async () => {
        // if (!svg) return;
        // const layerOutput = await getLayerOutput(model, layerName, inputImage);
        // renderVisualization(layerOutput);
    });

    // async function getLayerOutput(model: tf.LayersModel, layerName: string, input: tf.Tensor3D): Promise<number[][][]> {
    //     const layer = model.getLayer(layerName);
    //     const intermediateTensor = tf.tidy(() => {
    //         const inputTensor = input.expandDims(0);
    //         return (layer.apply(inputTensor) as tf.Tensor4D).squeeze([0]);
    //     });
    //     const output = await intermediateTensor.array() as number[][][];
    //     intermediateTensor.dispose();
    //     return output;
    // }

    // function renderVisualization(activations: number[][][]) {
    //     const numChannels = activations[0][0].length;
    //     const featureMapSize = activations.length;

    //     const svgSelection = d3.select(svg);
    //     svgSelection.selectAll("*").remove();

    //     const colorScale = d3.scaleSequential(d3.interpolateViridis)
    //         .domain([d3.min(activations.flat(2)), d3.max(activations.flat(2))]);

    //     const boxSize = Math.min((width - padding * 2) / numChannels, (height - padding * 2) / featureMapSize);

    //     for (let c = 0; c < numChannels; c++) {
    //         const group = svgSelection.append("g")
    //             .attr("transform", `translate(${padding + c * (boxSize + 5)}, ${padding})`);

    //         for (let y = 0; y < featureMapSize; y++) {
    //             for (let x = 0; x < featureMapSize; x++) {
    //                 const value = activations[y][x][c];
    //                 group.append("rect")
    //                     .attr("x", x * boxSize)
    //                     .attr("y", y * boxSize)
    //                     .attr("width", boxSize)
    //                     .attr("height", boxSize)
    //                     .attr("fill", colorScale(value))
    //                     .attr("stroke", "white")
    //                     .attr("stroke-width", 0.5);
    //             }
    //         }

    //         group.append("text")
    //             .attr("x", featureMapSize * boxSize / 2)
    //             .attr("y", featureMapSize * boxSize + 20)
    //             .attr("text-anchor", "middle")
    //             .text(`Channel ${c + 1}`);
    //     }
    // }
</script>

<!-- <svg bind:this={svg} {width} {height}></svg> -->
<div class="flex flex-col items-center gap-2 rounded-lg border bg-card p-2 text-card-foreground">
		<div class="h-6 w-6"></div>
	<div class="h-6 w-6"
	></div>
	{#each { length: layer.convlyr.filters } as _, nodeIndex (nodeIndex)}
		<div class="relative flex h-6 w-6 items-center justify-center">
			<HoverCard.Root>
				<HoverCard.Trigger>
                    <Bone class = "h-5 w-5 rounded-[0.15rem]" layerName={layerName} nodeIndex={nodeIndex}  rgb={true}/>
				</HoverCard.Trigger>
				<HoverCard.Content class="h-fit max-h-none w-fit max-w-none">
					<Bone class = "h-96 w-96 rounded-[0.15rem]" layerName={layerName} nodeIndex={nodeIndex} rgb={true}/>
					<!--<Losschart class="h-56 w-56 rounded-[0.15rem]" />-->
				</HoverCard.Content>
			</HoverCard.Root>
		</div>
	{/each}
</div>