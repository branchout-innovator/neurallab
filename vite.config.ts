import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import {searchForWorkspaceRoot} from 'vite';

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
	],
	server: {
		fs: {
		  allow: [
			// search up for workspace root
			searchForWorkspaceRoot(process.cwd()),
			// your custom rules
			'/neurallab/static',
		  ],
		},
	  },
});
