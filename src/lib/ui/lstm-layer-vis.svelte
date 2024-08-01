<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
	import type { DenseLayer, LSTMLayer, SequentialModel } from '$lib/structures';
	import { remToPx } from '$lib/utils';
	import Minus from 'lucide-svelte/icons/minus';
	import Plus from 'lucide-svelte/icons/plus';
	import { getContext, setContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import * as tf from '@tensorflow/tfjs';
	import { browser } from '$app/environment';
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';
	import Heatmap from './heatmap.svelte';
	import * as HoverCard from '$lib/components/ui/hover-card';
	import EnlargedHeatmap from './detailed-vis.svelte';
	import PredictionCurve from './prediction-curve.svelte';
	import isEqual from 'lodash.isequal';
	import ActivationColor from './activation-color.svelte';
    import LstmUnitVis from './lstm-unit-vis.svelte';
	import ConnectionsVis from './connections-vis.svelte';

	//export let index: number;
    export let index: number;
    export let timeSteps: number;
    export let units: number;
    const model: Writable<SequentialModel> = getContext('lstmmodel');

    function addUnit() {
        timeSteps += 1;
        ($model.layers[index] as LSTMLayer).timestep = timeSteps;
    }
    function removeUnit() {
        if (timeSteps == 1) return;
        timeSteps -= 1;
        ($model.layers[index] as LSTMLayer).timestep = timeSteps;
    }

    function getXPosEnd() {
        const nodeSpacing = remToPx(2);
        return [(Math.min(units, 10)-1)*nodeSpacing+nodeSpacing/2]
    }
    function getXPosStart() {
        const nodeSpacing = remToPx(2);
        return [nodeSpacing/2]
    }

</script>
<div
    class="flex w-fit flex-col overflow-x-auto overflow-y-hidden rounded-lg border p-1 text-sm"
>
    <div class="ml-auto mr-auto flex flex-row items-center">
        <Button variant="ghost" size="icon" class="h-8 w-8" on:click={addUnit}>
            <Plus class="h-4 w-4"></Plus>
        </Button>

        <Button variant="ghost" size="icon" class="h-8 w-8" on:click={removeUnit}>
            <Minus class="h-4 w-4"></Minus>
        </Button>
        <span class="ml-2 leading-none text-muted-foreground"
            >{timeSteps} Time Steps</span
        >
    </div>
	{#each { length: Math.min(10, timeSteps) } as _, nodeIndex (nodeIndex)}
		<LstmUnitVis units={units} />
        {#if nodeIndex != Math.min(10, timeSteps)-1}
        <ConnectionsVis 
        leftLayerHeights={getXPosEnd()}
        rightLayerHeights={getXPosStart()}
        canvasWidth={20}
        lstm={true}/>
        {/if}
	{/each}
    
</div>

