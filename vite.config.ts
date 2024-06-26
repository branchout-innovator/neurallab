import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		// viteStaticCopy({
		// 	targets: [
		// 		{
		// 			src: 'node_modules/@tensorflow/tfjs-backend-wasm/dist/*',
		// 			dest: 'static'
		// 		}
		// 	]
		// }),
		sveltekit()
	]
});
