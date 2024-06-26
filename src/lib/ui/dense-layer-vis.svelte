<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
	import type { DenseLayer, SequentialModel } from '$lib/structures';
	import { remToPx } from '$lib/utils';
	import Minus from 'lucide-svelte/icons/minus';
	import Plus from 'lucide-svelte/icons/plus';
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';

	export let layer: DenseLayer;
	export let index: number;

	const model: Writable<SequentialModel> = getContext('model');

	const setUnits = (units: number) => {
		($model.layers[index] as DenseLayer).units = units;
		// const nextLayer = $model.layers[index + 1] as DenseLayer;
		// if (nextLayer) {
		// 	nextLayer.inputShape = [layer.units];
		// }
	};
</script>

<div class="flex flex-col items-center gap-2 rounded-lg border bg-card p-2 text-card-foreground">
	<Button variant="ghost" size="icon" class="h-6 w-6" on:click={() => setUnits(layer.units + 1)}>
		<Plus class="h-4 w-4"></Plus>
	</Button>
	<Button
		variant="ghost"
		size="icon"
		class="h-6 w-6"
		on:click={() => setUnits(Math.max(layer.units - 1, 0))}
	>
		<Minus class="h-4 w-4"></Minus>
	</Button>
	{#each { length: layer.units } as _, nodeIndex (nodeIndex)}
		<div class="h-6 w-6 rounded-full bg-muted"></div>
	{/each}
</div>
