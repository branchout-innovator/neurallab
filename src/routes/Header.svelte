<script>
	import { page } from '$app/stores';
	import logo from '$lib/images/svelte-logo.svg';
	import github from '$lib/images/github.svg';
	import ThemeToggle from '$lib/ui/theme-toggle.svelte';
</script>

<header class="sticky top-0 flex h-14 items-center gap-4 border-b bg-background px-4 md:px-6">
	<nav
		class="hidden flex-col gap-6 text-lg font-medium md:flex md:flex-row md:items-center md:gap-5 md:text-sm lg:gap-6"
	>
		<a href="##" class="text-foreground transition-colors hover:text-foreground"> Dashboard </a>
	</nav>
	<div class="flex flex-1 items-center justify-between space-x-2 md:justify-end">
		<ThemeToggle></ThemeToggle>
	</div>
</header>
