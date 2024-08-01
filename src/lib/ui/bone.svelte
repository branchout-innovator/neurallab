<script lang="ts">
	import { Button } from '$lib/components/ui/button/index.js';
	import * as Popover from '$lib/components/ui/popover/index.js';
	import logo from '$lib/images/image0.png';
</script>

<Popover.Root portal={null}>
	<Popover.Trigger asChild let:builder>
		<Button builders={[builder]} variant="outline">Open</Button>
	</Popover.Trigger>
	<Popover.Content class="w-120">
		<div class = "w-80 float-left">
			<img alt="The project logo" src={logo} />
      </div>
      <div class = "w-80">
      </div>
      <div class = "w-80 float-right">
			<img alt="The project logo" src={logo} />
		</div>
	</Popover.Content>
</Popover.Root>
