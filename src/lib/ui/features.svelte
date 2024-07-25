<script lang="ts">
    import { getContext } from 'svelte';
    import type { Writable } from 'svelte/store';

    export let columnNames: string[] = ['x', 'y'];

    const csvColumnConfigs: Writable<{
        [key: string]: { isLabel: 'true' | 'false' };
    }> = getContext('csvColumnConfigs');

    $: inputFeatures = columnNames.length > 0 ? columnNames : ['x', 'y'];
</script>

<div class="flex flex-col items-center gap-2 rounded-lg border bg-card p-2 text-card-foreground">
    <h5 class="mb-8 font-bold text-sm">Input Features</h5>
    {#each inputFeatures.slice(0, -1) as feature}
        <div class="flex h-6 w-6 items-center justify-center">
            <span class="text-xs">{feature}</span>
        </div>
    {/each}
</div>