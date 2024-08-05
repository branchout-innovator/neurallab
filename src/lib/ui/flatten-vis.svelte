<script lang="ts">
	import { onMount } from 'svelte';
	import * as d3 from 'd3';
	import * as tf from '@tensorflow/tfjs';

	export let model: tf.LayersModel;
	export let layerName: string;
</script>
<div class="flex flex-col items-center gap-2 rounded-lg border p-2 bg-card text-card-foreground">
    <div class="relative flex h-20 w-6 items-center justify-center"> <span class="rotate-90">FLATTEN</span> </div>
	
</div>