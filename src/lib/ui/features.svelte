<script lang="ts">
	import { getContext, onMount } from 'svelte';
	import type { Writable } from 'svelte/store';
	export let columnNames: string[];
	export let currentExample: { xs: number[]; ys: number[] } | null;
	export let isImageDataset: boolean;
	export let numchannels: number;
	export let sampleImage: number[][][] = [[]];

	const csvColumnConfigs: Writable<{
		[key: string]: { isLabel: 'true' | 'false' };
	}> = getContext('csvColumnConfigs');

	$: labels = columnNames
		.filter((c) => $csvColumnConfigs[c]?.isLabel === 'false')
		.map((c, i) => ({ name: c, value: currentExample?.xs[i] }));

	// const splitChannels = (image) => {
	// 	const red = image.clone();
	// 	const green = image.clone();
	// 	const blue = image.clone();

	// 	red.setChannel(1, 0); //gb, rb, rg
	// 	red.setChannel(2, 0);

	// 	green.setChannel(0, 0);
	// 	green.setChannel(2, 0);

	// 	blue.setChannel(0, 0);
	// 	blue.setChannel(1, 0);

	// 	return { red, green, blue };
	// };

	let channelnames: string[];

	if(numchannels == 1){
		channelnames = ["black-white"]
	}
	else{
		channelnames = ["red", "green", "blue"]
	}

</script>

<div class="flex flex-col items-end gap-3 rounded-lg py-2 text-card-foreground">
	<h5 class="mb-9 text-sm">&nbsp;</h5>
	{#if isImageDataset}
		{#each channelnames as feature}
			<div class="flex flex-row items-center">
				<div
					class="flex items-center justify-start text-nowrap rounded bg-muted px-2 py-1 font-medium"
				>
					<span class="text-xs">{feature}</span>
				</div>
			</div>
		{/each}
	{:else}
		{#each labels as feature}
			<div class="flex flex-row items-center">
				<div
					class="flex items-center justify-start text-nowrap rounded bg-muted px-2 py-2 font-medium"
				>
					<span class="text-xs">{feature.name + (feature.value ? `: ${feature.value}` : '')}</span>
				</div>
			</div>
		{/each}
	{/if}
</div>
