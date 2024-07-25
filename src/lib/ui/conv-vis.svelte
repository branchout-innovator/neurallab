<script lang="ts">
    import { onMount } from 'svelte';
    import * as d3 from 'd3';
    import * as tf from '@tensorflow/tfjs';

    export let model: tf.LayersModel;
    export let layerName: string;
    export let inputImage: tf.Tensor3D;

    let svg: SVGSVGElement;
    let width = 600;
    let height = 400;
    let padding = 40;

    onMount(async () => {
        if (!svg) return;
        const layerOutput = await getLayerOutput(model, layerName, inputImage);
        renderVisualization(layerOutput);
    });

    async function getLayerOutput(model: tf.LayersModel, layerName: string, input: tf.Tensor3D): Promise<number[][][]> {
        const layer = model.getLayer(layerName);
        const intermediateTensor = tf.tidy(() => {
            const inputTensor = input.expandDims(0);
            return (layer.apply(inputTensor) as tf.Tensor4D).squeeze([0]);
        });
        const output = await intermediateTensor.array() as number[][][];
        intermediateTensor.dispose();
        return output;
    }

    function renderVisualization(activations: number[][][]) {
        const numChannels = activations[0][0].length;
        const featureMapSize = activations.length;

        const svgSelection = d3.select(svg);
        svgSelection.selectAll("*").remove();

        const colorScale = d3.scaleSequential(d3.interpolateViridis)
            .domain([d3.min(activations.flat(2)), d3.max(activations.flat(2))]);

        const boxSize = Math.min((width - padding * 2) / numChannels, (height - padding * 2) / featureMapSize);

        for (let c = 0; c < numChannels; c++) {
            const group = svgSelection.append("g")
                .attr("transform", `translate(${padding + c * (boxSize + 5)}, ${padding})`);

            for (let y = 0; y < featureMapSize; y++) {
                for (let x = 0; x < featureMapSize; x++) {
                    const value = activations[y][x][c];
                    group.append("rect")
                        .attr("x", x * boxSize)
                        .attr("y", y * boxSize)
                        .attr("width", boxSize)
                        .attr("height", boxSize)
                        .attr("fill", colorScale(value))
                        .attr("stroke", "white")
                        .attr("stroke-width", 0.5);
                }
            }

            group.append("text")
                .attr("x", featureMapSize * boxSize / 2)
                .attr("y", featureMapSize * boxSize + 20)
                .attr("text-anchor", "middle")
                .text(`Channel ${c + 1}`);
        }
    }
</script>

<svg bind:this={svg} {width} {height}></svg>

<style>
    svg {
        border: 1px solid #ccc;
    }
</style>