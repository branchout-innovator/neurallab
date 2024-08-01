<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
	import type { DenseLayer, SequentialModel } from '$lib/structures';
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

	//export let index: number;
    export let rate: number;
    
</script>

<div class="flex flex-col items-center gap-2 rounded-lg border p-2 bg-card text-card-foreground">
    <div class="relative flex h-20 w-6 items-center justify-center"> <span class="rotate-90">DROPOUT</span> </div>
	
    <div class="relative flex h-6 w-6 items-center justify-center">
        <div
            class="h-5 w-5 rounded-[0.15rem] justify-center"
        >
        <span>{rate}</span>
    </div>
		</div>
</div>
