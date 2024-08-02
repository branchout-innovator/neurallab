<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
    import * as tf from '@tensorflow/tfjs';
    import { writable, type Writable } from 'svelte/store';
    import { getContext, onMount, setContext } from 'svelte';
    import Losschart from '$lib/ui/losschart.svelte';
    import TrendingDown from 'lucide-svelte/icons/trending-down';
    import * as Card from '$lib/components/ui/card/index.js';
    import * as d3 from 'd3';
    import Minus from 'lucide-svelte/icons/minus';
	import Plus from 'lucide-svelte/icons/plus';
    import { getNodeYPositions, getNodeYPositionsInput, handler } from '$lib/ui/connections-vis';
    import DropoutVis from './dropout-vis.svelte';
    import {
        type ActivationIdentifier,
        createRNNModel,
		type DenseLayer,
		type DropoutLayer,
		type FlattenLayer,
		type Layer,
		type LSTMLayer,
		type SequentialModel,
    } from '$lib/structures';
	import DenseLayerVis from './dense-layer-vis.svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import ConnectionsVis from './connections-vis.svelte';
    export let index: number;

    const model: Writable<SequentialModel> = getContext('lstmmodel');
    let layer = ($model.layers[index] as DenseLayer);
    console.log(layer.units);
	
</script>
<div class="flex flex-col items-center gap-2 rounded-lg border bg-card p-2 text-card-foreground h-fit">
    <Button variant="ghost" size="icon" class="h-6 w-6">
        <Plus class="h-4 w-4"></Plus>
    </Button>
    <Button
        variant="ghost"
        size="icon"
        class="h-6 w-6"
    >
        <Minus class="h-4 w-4"></Minus>
    </Button>
    {#each { length: (Math.min(10,layer.units)) } as _, nodeIndex (nodeIndex)}
        <div class="relative flex h-6 w-6 items-center justify-center">
            <div
        style={`background-color: blue;`}
        class="h-5 w-5 rounded-[0.15rem]"
            ></div>
        </div>
    {/each}
</div>