import type { DenseLayer, Layer } from '$lib/structures';
import { remToPx } from '$lib/utils';

export const getNodeYPositions = (layer: Layer): number[] => {
	switch (layer.type) {
		case 'dense': {
			const denseLayer = layer as DenseLayer;
			const nodeSpacing = remToPx(2);
			return Array.from({ length: denseLayer.units }, (_, i) => i * nodeSpacing + nodeSpacing / 2);
		}
		// Add more cases for other layer types as needed
		default:
			throw new Error(`Unsupported layer type: ${layer.type}`);
	}
};
