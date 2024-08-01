<script>
	import { page } from '$app/stores';
	import logo from '$lib/images/svelte-logo.svg';
	import github from '$lib/images/github.svg';
	import ThemeToggle from '$lib/ui/theme-toggle.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import Plus from 'lucide-svelte/icons/plus';
	import Minus from 'lucide-svelte/icons/minus';
	import DrawerRoot from '$lib/components/ui/drawer/drawer.svelte';
	import DrawerHeader from '$lib/components/ui/drawer/drawer-header.svelte';
	import DrawerContent from '$lib/components/ui/drawer/drawer-content.svelte';
	import { Drawer as DrawerPrimitive } from "vaul-svelte";
	import DrawerFooter from '$lib/components/ui/drawer/drawer-footer.svelte';
	import DrawerTitle from '$lib/components/ui/drawer/drawer-title.svelte';
	import DrawerDescription from '$lib/components/ui/drawer/drawer-description.svelte';
	
	</script>
	
	<header class="flex h-14 items-center gap-4 border-b bg-background">
	<div class="container flex h-14 max-w-screen-2xl items-center">
	<nav
	class="hidden flex-col gap-6 text-lg font-medium md:flex md:flex-row md:items-center md:gap-5 md:text-sm lg:gap-6"
	>
	<a href="/" class="text-foreground transition-colors hover:text-foreground"> Dashboard </a>
	
	<div class = "fixed-top"><DrawerRoot>
	<DrawerPrimitive.Trigger asChild let:builder>
	<Button builders={[builder]} variant="ghost">About NeuralLab</Button>
	</DrawerPrimitive.Trigger>
	<DrawerContent>
	<div class="mx-auto w-full max-w-sm">
	<DrawerHeader>
	<DrawerTitle>About NeuralLab</DrawerTitle>
	<br>
	<DrawerDescription>NeuralLab is an interactive online platform designed to teach neural network concepts in artificial intelligence. Through its interactive, accessible, and easy to learn neural network, it offers clear, accessible lessons that cater to learners of all ages, enabling them to explore AI concepts at their own pace. The user-friendly interface also allows visitors to apply their ideas using custom datasets for real-time training, promoting both exploration and curiosity in AI.</DrawerDescription>
	</DrawerHeader>
	<div class="p-4 pb-0">
	<div class="flex items-center justify-center space-x-2">
	</div>
	<div class="mt-3 h-[120px]">
	</div>
	</div>
	</div>
	</DrawerContent>
	</DrawerRoot>
	</div>
	
	
	</nav>
	<div class="flex flex-1 items-center justify-between space-x-2 md:justify-end">
	<ThemeToggle></ThemeToggle>
	</div>
	</div>
	</header>