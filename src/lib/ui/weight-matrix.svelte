<script lang="ts">
	export let weights: number[][];

	function getColor(weight: number): string {
		const max = Math.max(...weights.flat().map(Math.abs));
		const normalizedWeight = weight / max;
		if (normalizedWeight > 0) {
			return `rgb(${255 - normalizedWeight * 255}, 255, ${255 - normalizedWeight * 255})`;
		} else {
			return `rgb(255, ${255 + normalizedWeight * 255}, ${255 + normalizedWeight * 255})`;
		}
	}
</script>

<div
	class="grid"
	style={`grid-template-rows: repeat(${weights.length}, 1fr); grid-template-columns: repeat(${weights[0]?.length || 0}, 1fr);`}
>
	{#each weights as weightRow, i (i)}
		{#each weightRow as weight, j (j)}
			<div
				style={`background-color: ${getColor(weight)}; color: ${Math.abs(weight) > 0.5 ? 'white' : 'black'}; padding: 4px; text-align: center;`}
			>
				{weight.toFixed(2)}
			</div>
		{/each}
	{/each}
</div>

<style>
	.grid {
		display: grid;
		gap: 1px;
		background-color: #ccc;
		padding: 1px;
	}
</style>