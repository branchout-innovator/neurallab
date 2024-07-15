<script lang="ts">
	import Heatmap from './heatmap.svelte';

	export let nodeIndex: number;
	export let layerName: string;
</script>

<div class="enlarged-heatmap">
	<Heatmap {nodeIndex} {layerName} class="h-56 w-56 rounded" />
</div>
