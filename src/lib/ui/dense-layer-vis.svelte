<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
	import type { DenseLayer } from '$lib/structures';
	import Plus from 'lucide-svelte/icons/plus';

	export let layer: DenseLayer;
</script>

<div class="flex flex-col items-center gap-2 rounded-lg border bg-card p-2 text-card-foreground">
	<Button variant="ghost" size="icon" class="h-6 w-6" on:click={() => layer.units++}>
		<Plus class="h-4 w-4"></Plus>
	</Button>
	{#each { length: layer.units } as _, nodeIndex (nodeIndex)}
		<div class="h-6 w-6 rounded-full bg-muted"></div>
	{/each}
</div>
