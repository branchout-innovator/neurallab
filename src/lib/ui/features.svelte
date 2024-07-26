<script lang="ts">
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';

	export let columnNames: string[];
	export let currentExample: { xs: number[]; ys: number[] } | null;

	const csvColumnConfigs: Writable<{
		[key: string]: { isLabel: 'true' | 'false' };
	}> = getContext('csvColumnConfigs');

	$: inputFeatures = columnNames
		.filter((c) => $csvColumnConfigs[c]?.isLabel === 'false')
		.map((c, i) => ({ name: c, value: currentExample?.xs[i] }));
</script>

<div class="flex flex-col items-end gap-2 rounded-lg py-2 text-card-foreground">
	<h5 class="mb-9 text-sm">&nbsp;</h5>
	{#each inputFeatures as feature}
		<div class="flex items-center justify-end rounded bg-muted px-2 py-1 font-medium">
			<span class="text-xs">{feature.name + (feature.value ? `: ${feature.value}` : '')}</span>
		</div>
	{/each}
</div>
