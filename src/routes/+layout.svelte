<script>
	import '../app.css';
	import Header from './header.svelte';
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from '$lib/components/ui/sonner';
</script>

<svelte:head>
	<title>NeuralLab</title>
</svelte:head>

<Header></Header>

<ModeWatcher />
<main>
	<slot></slot>
</main>

<footer></footer>

<Toaster />
