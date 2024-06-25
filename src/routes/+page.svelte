<script>
	import Counter from './Counter.svelte';
	import welcome from '$lib/images/svelte-welcome.webp';
	import welcome_fallback from '$lib/images/svelte-welcome.png';

	import * as Select from '$lib/components/ui/select';
	import Label from '$lib/components/ui/label/label.svelte';
</script>

<svelte:head>
	<title>Home</title>
	<meta name="description" content="Svelte demo app" />
</svelte:head>

<div class="container flex h-full max-w-screen-2xl flex-col gap-4 py-4">
	<div class="flex flex-col gap-2">
		<Label class="text-xs">Activation Function</Label>
		<Select.Root selected={{ value: 'relu', label: 'ReLU' }}>
			<Select.Trigger class="w-[180px]">
				<Select.Value></Select.Value>
			</Select.Trigger>
			<Select.Content>
				<Select.Item value="relu">ReLU</Select.Item>
				<Select.Item value="sigmoid">Sigmoid</Select.Item>
			</Select.Content>
		</Select.Root>
	</div>

	<div class="flex h-full flex-row items-center justify-center rounded-lg border p-10 text-sm">
		Neural net editor will go here
	</div>
</div>

<style>
	section {
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		flex: 0.6;
	}

	h1 {
		width: 100%;
	}

	.welcome {
		display: block;
		position: relative;
		width: 100%;
		height: 0;
		padding: 0 0 calc(100% * 495 / 2048) 0;
	}

	.welcome img {
		position: absolute;
		width: 100%;
		height: 100%;
		top: 0;
		display: block;
	}
</style>
