<script lang="ts">
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';

	export let columnNames: string[];
	export let currentExample: { xs: number[]; ys: number[] } | null;

	const csvColumnConfigs: Writable<{
		[key: string]: { isLabel: 'true' | 'false' };
	}> = getContext('csvColumnConfigs');

	$: labels = columnNames
		.filter((c) => $csvColumnConfigs[c]?.isLabel === 'true')
		.map((c, i) => ({ name: c, value: currentExample?.ys[i] }));
</script>

<div class="flex flex-col items-start gap-2 rounded-lg py-2 text-card-foreground">
	<h5 class="mb-9 text-sm">&nbsp;</h5>
	{#each labels as feature}
		<div class="flex flex-row items-center">
			<div class="w-6 border-t border-muted-foreground"></div>
			<div class="flex items-center justify-start rounded bg-muted px-2 py-1 font-medium">
				<span class="text-xs">{feature.name + (feature.value ? `: ${feature.value}` : '')}</span>
			</div>
		</div>
	{/each}
</div>
