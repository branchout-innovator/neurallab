<script>
	import '../app.css';
	import Header from './header.svelte';
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

<footer></footer>
