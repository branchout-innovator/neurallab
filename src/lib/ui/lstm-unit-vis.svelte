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
    export let units: number;



	const sampleDomain: Writable<{ x: [number, number]; y: [number, number] }> =
		getContext('sampleDomain');
	
	let mapComponent: EnlargedHeatmap;
    export function zoom() {
        mapComponent.setZoom();
    }
    
</script>

<div class="flex flex-row items-center gap-2 rounded-lg border bg-card p-2 text-card-foreground">
	<span>{units} Units</span>
	{#each { length: Math.min(10, units) } as _, nodeIndex (nodeIndex)}
		<div class="relative flex h-6 w-6 items-center justify-center">
			<HoverCard.Root>
				<HoverCard.Trigger>
					<div
                        style={`background-color: blue;`}
                        class="h-5 w-5 rounded-[0.15rem]"
                    ></div>
				</HoverCard.Trigger>
				<HoverCard.Content class="h-fit max-h-none w-fit max-w-none">
				</HoverCard.Content>
			</HoverCard.Root>
		</div>
	{/each}
</div>
