<script lang="ts">
	import { onMount } from 'svelte';
	import * as d3 from 'd3';
	import * as tf from '@tensorflow/tfjs';

	export let model: tf.LayersModel;
	export let layerName: string;
</script>

<div>Flatten Layer</div>
