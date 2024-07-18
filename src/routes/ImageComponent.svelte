<script>
    export let href = "";
    export let title = undefined;
    export let text = "";
    let darkhref = href + "dark.png";
    href+=".png";
  </script>
  
<img
    class="hidden dark:block"
    src={darkhref}
    alt="dark-mode"
/>
<img
    class="mb-4 block dark:hidden"
    src={href}
    alt="light-mode"
/>