<!--ignore this
<script lang="ts">
    import { onMount } from 'svelte';
    import * as d3 from 'd3';
    import * as tf from '@tensorflow/tfjs';

    export let model: tf.LayersModel;
    export let convLayerName: string;
    export let reluLayerName: string;
    export let inputImage: tf.Tensor3D;

    let svg: SVGSVGElement;
    let width = 800;
    let height = 400;
    let padding = 40;

    onMount(async () => {
        if (!svg) return;
        
        const convOutput = await getLayerOutput(model, convLayerName, inputImage);
        const reluOutput = await getLayerOutput(model, reluLayerName, inputImage);

        renderVisualization(convOutput, reluOutput);
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
-->
