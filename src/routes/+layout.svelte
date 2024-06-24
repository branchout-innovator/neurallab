<script>
	import '../app.css';
	import Header from './Header.svelte';
	import { ModeWatcher } from 'mode-watcher';
</script>

<svelte:head>
	<title>NeuralLab</title>
</svelte:head>

<Header></Header>

<ModeWatcher />
<main>
	<slot></slot>
</main>

<footer>
	<p>visit <a href="https://kit.svelte.dev">kit.svelte.dev</a> to learn SvelteKit</p>
</footer>
