import type { Conv2DLayer, DenseLayer, Layer, LSTMLayer, MaxPoolingLayer } from '$lib/structures';
import { remToPx } from '$lib/utils';

export const getNodeYPositions = (layer: Layer): number[] => {
	switch (layer.type) {
		case 'dense': {
			const denseLayer = layer as DenseLayer;
			let units;
			if (denseLayer.lstm) {
				units = Math.min(10, denseLayer.units);
			}
			else {
				units = denseLayer.units;
			}
			const nodeSpacing = remToPx(2.7);
			return Array.from({ length: units }, (_, i) => i * nodeSpacing + nodeSpacing / 2);
		}
		case 'flatten': {
			return [10];
		}
		case 'maxpooling': {
			const denseLayer = layer as MaxPoolingLayer;
			const nodeSpacing = remToPx(2.7);
			return Array.from({ length: denseLayer.convlyr.filters }, (_, i) => i * nodeSpacing + nodeSpacing / 2);
		}
		case 'conv2d': {
			const denseLayer = layer as Conv2DLayer;
			const nodeSpacing = remToPx(2.7);
			return Array.from({ length: denseLayer.filters }, (_, i) => i * nodeSpacing + nodeSpacing / 2);
		}
		case 'dropout': {
			return [10];
		}
		case 'lstm': {
			const lstmLayer = layer as LSTMLayer;
			const nodeSpacing = remToPx(2.7); //3.8?
			const offset = 28;
			return Array.from({ length: Math.min(lstmLayer.timestep, 10) }, (_, i) => i * nodeSpacing + nodeSpacing / 2-offset);
		}
		// Add more cases for other layer types as needed
		default:
			throw new Error(`TODO: implement layer type: ${layer.type} in connections-vis.ts`);
	}
};

export const handler = (pos: number[], times: number): number[] => {
	if (times < pos.length) return pos;
	let newarr = [];
	for (let i = 0; i < times; i++) {
		newarr.push(pos[0]);
	}
	return newarr;
}

export const getNodeYPositionsInput = (features: number): number[] => {
	const nodeSpacing = remToPx(2.7);
	return Array.from({ length: features }, (_, i) => i * nodeSpacing + nodeSpacing / 2);
};
