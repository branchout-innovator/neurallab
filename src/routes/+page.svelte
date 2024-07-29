<script lang="ts">
	import * as Select from '$lib/components/ui/select';
	import Label from '$lib/components/ui/label/label.svelte';
	import {
		createTFModel,
		loadUploadedCsv,
		type SampledOutputs,
		updateSampledOutputs,
		type ActivationIdentifier,
		type DenseLayer,
		type Layer,
		type SequentialModel,
		updateSampledOutputs1D,
		updateSampledOutputsSingle,
		type NestedArray,
		type LayerType,
		type Conv2DLayer,
		type MaxPoolingLayer,
		type FlattenLayer
	} from '$lib/structures';
	import * as tf from '@tensorflow/tfjs';
	import { onMount, setContext, SvelteComponent, getContext } from 'svelte';
	import DenseLayerVis from '$lib/ui/dense-layer-vis.svelte';
	import ConvVis from '$lib/ui/conv-vis.svelte';
	import MaxPoolingVis from '$lib/ui/max-pooling-vis.svelte';
	import FlattenVis from '$lib/ui/flatten-vis.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import Plus from 'lucide-svelte/icons/plus';
	import Minus from 'lucide-svelte/icons/minus';
	import Brain from 'lucide-svelte/icons/brain';
	import CirclePause from 'lucide-svelte/icons/circle-pause';
	import Activity from 'lucide-svelte/icons/activity';
	import RefreshCw from 'lucide-svelte/icons/refresh-cw';
	import TrendingDown from 'lucide-svelte/icons/trending-down';
	import { writable, type Writable } from 'svelte/store';
	import Input from '$lib/components/ui/input/input.svelte';
	import { toast } from 'svelte-sonner';
	import ConnectionsVis from '$lib/ui/connections-vis.svelte';
	import { getNodeYPositions, getNodeYPositionsInput } from '$lib/ui/connections-vis';
	import { Switch } from '$lib/components/ui/switch/index.js';
	import { browser } from '$app/environment';
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';
	import FileInput from '$lib/components/ui/file-input/file-input.svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import { buttonVariants } from '$lib/components/ui/button';
	import * as Resizable from '$lib/components/ui/resizable';
	import * as d3 from 'd3';
	import * as Table from '$lib/components/ui/table';
	import * as RadioGroup from '$lib/components/ui/radio-group';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu/index.js';
	import SvelteMarkdown from 'svelte-markdown';
	import ResizableHandle from '$lib/components/ui/resizable/resizable-handle.svelte';
	import isEqual from 'lodash.isequal';
	import ImageComponent from './ImageComponent.svelte';
	import mark from '$lib/articles/article1.md?raw';
	import mark2 from '$lib/articles/article2.md?raw';
	import mark3 from '$lib/articles/article3.md?raw';
	import mark4 from '$lib/articles/article4.md?raw';
	import mark5 from '$lib/articles/article5.md?raw';
	import ChevronLeft from 'lucide-svelte/icons/chevron-left';
	import ChevronRight from 'lucide-svelte/icons/chevron-right';
	import * as Tabs from '$lib/components/ui/tabs/index.js';
	import * as Card from '$lib/components/ui/card/index.js';
	import * as Command from '$lib/components/ui/command';
	import ThemeToggle from '$lib/ui/theme-toggle.svelte';
	import Features from '$lib/ui/features.svelte';
	import Labels from '$lib/ui/labels.svelte';
	import { Progress } from '$lib/components/ui/progress';
	import Slider from '@bulatdashiev/svelte-slider';
	import Losschart from '$lib/ui/losschart.svelte';
	import * as HoverCard from '$lib/components/ui/hover-card';
	import LLM from '$lib/ui/llm.svelte';

	let isImageDataset = false;
	let outputColumn = '';

	$: {
		if (datasetUploadFiles && datasetUploadFiles.length > 0) {
			(async () => {
				$csvColumnConfigs = {};
				const { columns } = d3.csvParse(await datasetUploadFiles[0].text());
				isImageDataset = columns.length > 50;
				console.log('image? ' + isImageDataset);
				for (const column of columns) {
					$csvColumnConfigs[column] = {
						isLabel: 'false'
					};
				}
			})();
		}
	}

	let value = 0;
	let losschart: Losschart;
	let currentloss = 0;
	let domain = [2, 8];
	let range = [2, 8];
	onMount(() => {
		const interval = setInterval(
			() =>
				(value =
					(100 * document.getElementById('article')!.scrollTop) /
					(document.getElementById('article')!.scrollHeight -
						document.getElementById('article')!.clientHeight)),
			100
		);
		return () => clearInterval(interval);
	});

	const layerComponents: Record<string, typeof SvelteComponent> = {
		dense: DenseLayerVis as typeof SvelteComponent,
		conv2d: ConvVis as typeof SvelteComponent,
		maxpooling: MaxPoolingVis as typeof SvelteComponent,
		flatten: FlattenVis as typeof SvelteComponent
	};

	let selectedActivation = { value: 'relu' as ActivationIdentifier, label: 'ReLU' };
	let epochs = 1000;
	let alpha = 0.01;

	const model = writable<SequentialModel>({
		layers: [
			{
				type: 'dense',
				units: 10,
				inputShape: [2]
			} as DenseLayer,
			{
				type: 'dense',
				units: 10
			} as DenseLayer,
			{
				type: 'dense',
				units: 1
			} as DenseLayer
		],
		loss: 'meanSquaredError',
		optimizer: 'adam',
		learningRate: 0.001
	});

	/*function updateModel() {
        model.optimizer.learningRate = alpha;
        tfModel = createTFModel(model);
    }

	onMount(() => {
        updateModel();
    });*/

	setContext('model', model);

	const addLayer = (type: LayerType) => {
		const lastLayer = $model.layers[$model.layers.length - 1] as DenseLayer;
		// if (lastLayer.type === 'dense')
		// 	($model.layers[$model.layers.length - 1] as DenseLayer).activation =
		// 		selectedActivation.value as ActivationIdentifier;
		let layer: Layer;
		switch (type) {
			case 'dense': {
				layer = {
					type: 'dense',
					units: 1
				} as DenseLayer;
				break;
			}
			case 'conv2d': {
				layer = {
					type: 'conv2d',
					filters: 8,
					kernelSize: [5, 5],
					strides: [1, 1]
				} as Conv2DLayer;
				break;
			}
			case 'maxpooling': {
				layer = {
					type: 'maxpooling',
					poolSize: [2, 2],
					strides: [2, 2]
				} as MaxPoolingLayer;
				break;
			}
			case 'flatten': {
				layer = {
					type: 'flatten'
				} as FlattenLayer;
			}
		}
		$model.layers = [...$model.layers, layer];
		if ($model.layers.length === 1) {
			$model.layers[0].inputShape = [$featureCount];
		}
		console.log($model.layers);
		refreshModel();
		updateTFModel($model);
	};

	const removeLayer = () => {
		if ($model.layers.length > 0) {
			$model.layers = [...$model.layers.slice(0, $model.layers.length - 1)];
			const lastLayer = $model.layers[$model.layers.length - 1];
			if (lastLayer?.type === 'dense') {
				($model.layers[$model.layers.length - 1] as DenseLayer).activation = undefined;
			}
		}
		console.log($model);
		updateTFModel($model);
	};

	async function generateData(numPoints: number, range: { max: number; min: number }) {
		const xs = [];
		const ys = [];
		for (let i = 0; i < numPoints; i++) {
			// const x = Math.random() * (range.max - range.min) + range.min;
			const x = range.min + (i / numPoints) * (range.max - range.min);
			xs.push(x);
			ys.push(x * x);
		}
		// Normalize the data
		const xTensor = tf.tensor2d(xs, [numPoints, 1]);
		const yTensor = tf.tensor2d(ys, [numPoints, 1]);
		// const xMin = xTensor.min();
		// const xMax = xTensor.max();
		// const yMin = yTensor.min();
		// const yMax = yTensor.max();

		// const normalizedXs = xTensor.sub(xMin).div(xMax.sub(xMin));
		// const normalizedYs = yTensor.sub(yMin).div(yMax.sub(yMin));

		return { xs: await xTensor.array(), ys: await yTensor.array() };
	}

	async function generateDataset(): Promise<tf.data.Dataset<tf.TensorContainer>> {
		const { xs, ys } = await generateData(1000, { min: -10, max: 10 });
		const xDataset = tf.data.array(xs);
		const yDataset = tf.data.array(ys);
		const xyDataset = tf.data.zip({ xs: xDataset, ys: yDataset }).batch(64).shuffle(4);
		return xyDataset;
	}

	let currentEpoch = 0;

	let currentExample: { xs: number[]; ys: number[] } | null;

	const sampleOutputs = async () => {
		if (!tfModel) return;
		if (!$model.layers[0]) return;
		if (isEqual($model.layers[0]?.inputShape, [1]))
			$sampledOutputs = await updateSampledOutputs1D(tfModel, SAMPLE_DENSITY_1D, $sampleDomain.x);
		else if (isEqual($model.layers[0]?.inputShape, [2])) {
			$sampledOutputs = await updateSampledOutputs(
				tfModel,
				SAMPLE_DENSITY_2D,
				$sampleDomain.x,
				$sampleDomain.y
			);
		}
		else {
			$dataset.take(1).forEachAsync(async (e) => {
				if (!tfModel) return;
				currentExample = e as { xs: number[]; ys: number[] };
				if (currentExample != null)
					$sampledOutputs = await updateSampledOutputsSingle(tfModel, currentExample.xs);
			});
		}
	};

	let isTraining = false;

	const refreshModel = () => {
		if (isTraining) isTraining = false;
		if (tfModel) tfModel!.stopTraining = true;
		currentEpoch = 0;
		tfModel = createTFModel($model);
		currentloss = 0;
		sampleOutputs();
		losschart?.clear();
		if (losscardVisible == 1) {
			displayLoss();
		}
	};

	const trainModel = async () => {
		if (!tfModel) return;
		if (isTraining) {
			isTraining = false;
			tfModel.stopTraining = true;
			return;
		}
		// The model needs to be "big enough" to benefit from GPU acceleration
		// So with the small models in the Tensorflow playground its actually faster to use CPU
		// await tf.setBackend('webgl');
		console.log(tf.getBackend());

		const data = $dataset;

		// toast.loading(`Training for ${epochs} epochs...`);

		// Train the model using the data.
		try {
			isTraining = true;
			const history = await tfModel.fitDataset(data.batch(64), {
				epochs: 1000000,
				callbacks: {
					async onEpochEnd(epoch, logs) {
						if (!tfModel) return;
						if (!isTraining) return;
						currentEpoch++;
						if (epoch % 5 === 0) tfModel = tfModel;
						try {
							await sampleOutputs();
							sampleDomain.set({
								x: [domain[0] - 5, domain[1] - 5],
								y: [range[0] - 5, range[1] - 5]
							});
						} catch (e) {
							console.error('Error while sampling outputs: ', e);
						}
						if (logs) {
							currentloss = Math.round(logs.loss * 1000) / 1000;
							if (losschart) {
								losschart.updateGraph(logs.loss);
							}
						}

						// set tfModel.stopTraining = true to stop training
					}
				}
			});
			//console.log(tfModel);
			//console.log($model);

			// // Make predictions and denormalize
			// const input = tf.tensor2d([Number(testPred)], [1, 1]);
			// // const normalizedInput = input.sub(data.xMin).div(data.xMax.sub(data.xMin));
			// const prediction = tfModel.predict(input) as tf.Tensor; // Assert that this is a single tensor
			// // const denormalizedPrediction = prediction.mul(data.yMax.sub(data.yMin)).add(data.yMin);
			// prediction.print(); // Should print a value close to 4 (2^2)

			// Open the browser devtools to see the output
			// toast.success(`Training complete!`);
			tfModel = tfModel;
		} catch (e) {
			console.error(e);
			toast.error(`Error while training: ${e}. Make sure the last layer has only 1 node.`, {
				duration: 100000
			});
		}
	};

	const updateActivations = () => {
		refreshModel();
		for (let i = 0; i < $model.layers.length; i++) {
			const layer = $model.layers[i];
			if (layer.type === 'dense' && i < $model.layers.length - 1) {
				(layer as DenseLayer).activation = selectedActivation.value;
			}
		}
		updateTFModel($model);
	};

	$: {
		selectedActivation.value;
		updateActivations();
	}

	function getWeightsBetweenLayers(model: tf.Sequential, layerIndex: number): tf.Tensor | null {
		const weights = model.layers[layerIndex].getWeights();
		if (!weights) return null;
		return weights[0]; // Assuming the first tensor in weights array is the weight matrix
	}
	let canvasWidth = 150;

	let tfModel: tf.Sequential | undefined = undefined;
	setContext('getTfModel', () => {
		return tfModel;
	});

	const updateTFModel = async (model: SequentialModel) => {
		if (model.layers.length === 0) return;
		refreshModel();
		if (!tfModel) {
			tfModel = createTFModel(model);
		} else {
			// TODO: preserve old weights when creating model with new structure
			const newModel = createTFModel(model);
			tfModel = newModel;
		}
		if (browser) {
			try {
				await sampleOutputs();
			} catch (e) {
				console.error('Error when sampling outputs: ', e);
			}
		}
	};

	$: {
		// updateTFModel($model);
	}

	// $: predictedVal = tfModel?.predict(tf.tensor2d([Number(testPred)], [1, 1]));
	$: predictedVal = 0;

	let testPred = 2;

	let useGPU = false;
	$: {
		if (browser) tf.setBackend(useGPU ? 'webgl' : 'cpu');
	}

	let datasetUploadFiles: FileList;

	let dataset: Writable<tf.data.Dataset<tf.TensorContainer>> = writable();
	setContext('dataset', dataset);

	let csvColumnConfigs: Writable<{
		[key: string]: { isLabel: 'true' | 'false' };
	}> = writable({});
	setContext('csvColumnConfigs', csvColumnConfigs);

	$: {
		(async () => {
			if (datasetUploadFiles && datasetUploadFiles.length > 0) {
				const { columns } = d3.csvParse(await datasetUploadFiles[0].text());
				$csvColumnConfigs = {};
				for (const column of columns) {
					$csvColumnConfigs[column] = {
						isLabel: 'false'
					};
				}
			}
		})();
	}

	let hasLabel = false;
	let featureCount: Writable<number> = writable(-1);

	const updateDataset = async (
		datasetUploadFiles: FileList,
		csvColumnConfigs: {
			[key: string]: { isLabel: 'true' | 'false' };
		}
	) => {
		if (datasetUploadFiles && datasetUploadFiles.length) {
			const config: {
				[key: string]: tf.data.ColumnConfig;
			} = {};
			hasLabel = false;
			$featureCount = 0;

			if (isImageDataset) {
				for (const column in $csvColumnConfigs) {
					if (column === outputColumn) {
						config[column] = { isLabel: true };
						hasLabel = true;
					} else if (!column.includes('x')) {
						config[column] = { isLabel: false };
						$featureCount++;
					}
				}
			} else {
				for (const column in $csvColumnConfigs) {
					const isLabel = $csvColumnConfigs[column].isLabel === 'true';
					config[column] = { isLabel };
					if (isLabel) {
						hasLabel = true;
					} else {
						$featureCount++;
					}
				}
			}

			if (hasLabel) {
				const result = await loadUploadedCsv(datasetUploadFiles[0], config);
				$dataset = result.dataset;
				columnNames = result.columnNames;
				if ($model.layers[0]) {
					$model.layers[0].inputShape = [$featureCount];
					console.log('features: ', $featureCount);
					await updateTFModel($model);
				}
				await sampleOutputs();
			}
		}
	};

	$: updateDataset(datasetUploadFiles, $csvColumnConfigs);

	function pageLeft() {
		changePage(-1);
	}
	function pageRight() {
		changePage(1);
	}
	let articletitle = [
		[
			'What are Neural Networks? (Basics of Neural Networks)',
			'What are Activation Functions?',
			'What are Loss Functions? (Neural Nets)',
			'Optimization Algorithms'
		],
		[
			'Basics of CNNs',
			'Convolutional Layers',
			'Pooling Layers',
			'LeNet and advanced CNN architectures'
		]
	];
	let subtitles = [
		'Fundamentals of Neural Networks',
		'Convolutional Neural Networks (CNNs)',
		'Recurrent Neural Networks (RNNs)'
	];
	let pagetext = [mark, mark2, mark3, mark4, mark5];
	$: source = pagetext[Number(position)];
	function changePage(d: number) {
		let pageNum = Number(position);
		if ((pageNum != 0 || d != -1) && (pageNum != pagetext.length - 1 || d != 1)) {
			pageNum += d;
		}
		position = String(pageNum);
		source = pagetext[pageNum];
	}

	let position = '0';
	let sampledOutputs = writable<SampledOutputs<NestedArray>>({});
	setContext('sampledOutputs', sampledOutputs);

	const SAMPLE_DENSITY_2D = 10;
	const SAMPLE_DENSITY_1D = 10;
	let sampleDomain: Writable<{ x: [number, number]; y: [number, number] }> = writable({
		x: [domain[0], domain[1]],
		y: [range[0], range[1]]
	});
	setContext('sampleDomain', sampleDomain);

	$: is1D = isEqual($model.layers[0]?.inputShape, [1]);
	$: is2D = isEqual($model.layers[0]?.inputShape, [2]);

	$: {
		$sampleDomain.x = (is1D ? [-11, 11] : domain) as [number, number];
		$sampleDomain.y = (is1D ? [-10, 120] : range) as [number, number];
	}

	let columnNames: string[] = [];

	const loadSampleDataset = async (url: string, numFeatures: number) => {
		let blob = await fetch(url).then((r) => r.blob());
		$csvColumnConfigs = {
			x: {
				isLabel: 'false'
			},
			y: {
				isLabel: 'false'
			},
			inside_circle: {
				isLabel: 'true'
			}
		};
		loadUploadedCsv(blob, {
			inside_circle: { isLabel: true }
		}).then((d) => {
			$dataset = d.dataset;
			columnNames = d.columnNames;
			sampleOutputs();
		});
		$featureCount = numFeatures;
		if ($model.layers[0]) {
			$model.layers[0].inputShape = [numFeatures];
			console.log('features: ', numFeatures);
			updateTFModel($model);
		}
	};

	$: {
		sampleDomain;
		sampleOutputs();
	}

	onMount(async () => {
		loadSampleDataset('/circle_dataset.csv', 2);
	});
	function addLength(accumulator: number, a: string[]) {
		return accumulator + a.length;
	}
	$: updateTFModel($model);
	let losscardVisible = 0;
	function displayLoss() {
		losscardVisible = 1-losscardVisible;
		d3.select("#losscard").style("visibility", ["hidden", "visible"][losscardVisible]);
	}
</script>

<svelte:head>
	<title>NeuralLab</title>
	<meta name="description" content="Design and visualize neural networks in your browser." />
</svelte:head>
<!--<div class="container flex h-full max-w-full flex-row gap-4">-->
<Resizable.PaneGroup direction="horizontal" class="container flex h-full max-w-full flex-row gap-4">
	<Resizable.Pane defaultSize={25}>
		<div class="container flex h-full w-full flex-col overflow-y-hidden px-0 py-4">
			<div class="h-1/8 container flex w-full flex-row items-end">
				<div class="flex h-full w-1/3">
					<Button variant="outline" class="ml-auto h-full" size="icon" on:click={pageLeft}
						>&lt;</Button
					>
				</div>
				<div class="w-1/3">
					<DropdownMenu.Root>
						<DropdownMenu.Trigger asChild let:builder>
							<Button variant="outline" class="w-full" builders={[builder]}>Article Sections</Button
							>
						</DropdownMenu.Trigger>
						<DropdownMenu.Content>
							<DropdownMenu.Label>Click for Pages</DropdownMenu.Label>
							<DropdownMenu.Separator />
							{#each articletitle as subcategory, j}
								<DropdownMenu.Sub>
									<DropdownMenu.SubTrigger>
										<span>{subtitles[j]}</span>
									</DropdownMenu.SubTrigger>
									<DropdownMenu.SubContent class="w-full">
										<DropdownMenu.RadioGroup bind:value={position}>
											{#each subcategory as title, i}
												<DropdownMenu.RadioItem
													value={String(i + articletitle.slice(0, j).reduce(addLength, 0))}
													>{title}</DropdownMenu.RadioItem
												>
											{/each}
										</DropdownMenu.RadioGroup>
									</DropdownMenu.SubContent>
								</DropdownMenu.Sub>
							{/each}
						</DropdownMenu.Content>
					</DropdownMenu.Root>
				</div>
				<div class="flex h-full w-1/3">
					<Button variant="outline" class="mr-auto h-full" size="icon" on:click={pageRight}
						>&gt;</Button
					>
				</div>
			</div>
			<span class="inline-block h-8 w-4" />
			<Progress {value} />
			<div id="article" class="flex w-full overflow-y-auto">
				<div class="w-full p-4">
					<h2
						class="scroll-m-20 border-b pb-2 text-center text-2xl font-semibold tracking-tight transition-colors first:mt-0"
					>
						{articletitle.flat()[Number(position)]}
					</h2>
					<span class="inline-block h-4 w-4"></span>
					<SvelteMarkdown {source} renderers={{ image: ImageComponent }} />
				</div>
			</div>
		</div>
	</Resizable.Pane>
	<Resizable.Handle withHandle />
	<Resizable.Pane defaultSize={60} class="p-4">
		<div class="flex h-full max-w-full flex-grow flex-col gap-4 overflow-x-hidden py-4">
			<!-- Controls (header) -->
			<Tabs.Root value="NL" class="h-auto w-full">
				<Tabs.List class="grid w-full grid-cols-3">
					<Tabs.Trigger value="NL">NeuralLab</Tabs.Trigger>
					<Tabs.Trigger value="settings">Settings</Tabs.Trigger>
					<Tabs.Trigger value="LLM">LLM</Tabs.Trigger>
				</Tabs.List>
				<Tabs.Content value="settings" class="h-full">
					<Card.Root class="h-full">
						<Card.Header>
							<Card.Title>Settings</Card.Title>
							<div class="float-left">Make changes to your settings here.</div>
						</Card.Header>
						<Card.Content class="space-y-3">
							<div class="flex flex-1 items-start space-x-2">
								<Label class="flex gap-2 text-xs">Choose Mode Here</Label>
							</div>
							<div>
								<ThemeToggle></ThemeToggle>
							</div>
							<br />
							<div class="grid w-1/3 grid-cols-2 items-center">
								<div>
									<Label class="flex gap-2 text-xs">Hardware</Label>
									<Tooltip.Root>
										<Tooltip.Trigger asChild>
											<div class="flex h-9 flex-row flex-nowrap items-center space-x-2">
												<Label for="hardware-backend">CPU</Label>
												<Switch id="hardware-backend" bind:checked={useGPU} />
												<Label for="hardware-backend">GPU</Label>
											</div>
										</Tooltip.Trigger>
										<Tooltip.Content class="max-w-52">
											GPU is recommended for large models but slower for small models.
										</Tooltip.Content>
									</Tooltip.Root>
								</div>
								<div class="flex flex-col gap-2">
									<Label class="flex gap-2 text-xs">&#945; Alpha Level</Label>
									<Input
										type="number"
										bind:value={alpha}
										placeholder="0.01"
										min={1}
										max={5}
										class="w-24"
									/>
								</div>
							</div>
							<br />
							<div class="space-y-1">
								<Label class="flex gap-2 text-xs">Choose Dataset</Label>
								<Dialog.Root>
									<Dialog.Trigger class={buttonVariants({ variant: 'outline' })}
										>Upload CSV</Dialog.Trigger
									>
									<Dialog.Content>
										<Dialog.Header>
											<Dialog.Title>Upload CSV Dataset</Dialog.Title>
											<Dialog.Description class="flex flex-col gap-2">
												<p>Upload a dataset from a .csv file.</p>
												<div class="flex flex-col">
													<FileInput
														id="dataset-upload"
														class="w-64"
														bind:files={datasetUploadFiles}
													/>
												</div>
												<div class="flex flex-col">
													<Input
														placeholder = "Enter Image Width"
														class="w-64"
													/>
												</div>
												<div class="flex flex-col">
													<Input
														placeholder = "Enter Image Height"
														class="w-64"
													/>
												</div>
												{#if Object.entries($csvColumnConfigs).length > 0}
													{#if isImageDataset}
														<div class="flex flex-col gap-2">
															<Label>Select Output Column</Label>
															<Command.Root>
																<Command.Input placeholder="Search output column..." />
																<Command.List>
																	<Command.Empty>No results found.</Command.Empty>
																	{#each Object.keys($csvColumnConfigs).filter((col) => !col.includes('x')) as column}
																		<Command.Item onSelect={() => (outputColumn = column)}>
																			{column}
																		</Command.Item>
																	{/each}
																</Command.List>
															</Command.Root>
														</div>
													{:else}
														<Table.Root>
															<Table.Header>
																<Table.Row>
																	<Table.Head class="flex-grow">Column Name</Table.Head>
																</Table.Row>
															</Table.Header>
															{#each Object.entries($csvColumnConfigs) as [column, config] (column)}
																<Table.Row>
																	<Table.Cell class="font-medium">{column}</Table.Cell>
																	<RadioGroup.Root
																		bind:value={$csvColumnConfigs[column].isLabel}
																		asChild
																	>
																		<Table.Cell>
																			<div class="flex items-center space-x-2">
																				<RadioGroup.Item value="false" id={`feature-${column}`}
																				></RadioGroup.Item>
																				<Label for={`feature-${column}`}>Input</Label>
																			</div>
																		</Table.Cell>
																		<Table.Cell>
																			<div class="flex items-center space-x-2">
																				<RadioGroup.Item value="true" id={`label-${column}`}
																				></RadioGroup.Item>
																				<Label for={`label-${column}`}>Output</Label>
																			</div>
																		</Table.Cell>
																	</RadioGroup.Root>
																</Table.Row>
															{/each}
														</Table.Root>
														{#if !hasLabel}
															<p class="font-medium text-foreground">
																Choose at least one column to use as output.
															</p>
														{/if}
													{/if}
												{/if}
											</Dialog.Description>
										</Dialog.Header>
									</Dialog.Content>
								</Dialog.Root>
							</div>
							<br />
							<div>
								Domain: left bound: {domain[0] - 5}
								right bound: {domain[1] - 5}
								<Slider max="10" step="1" bind:value={domain} range slider />
								Range: bottom bound: {range[0] - 5}
								top bound: {range[1] - 5}
								<Slider max="10" step="1" bind:value={range} range slider />
								<!-- <Button on:click={heatmap.changeZoom(domain, range)}>Change Axes</Button> -->
							</div>
							<!-- <div class="flex flex-col gap-2">
								<Label class="flex gap-2 text-xs">Input</Label>
								<Input type="number" bind:value={testPred} placeholder="2" class="w-24" />
							</div>
							<div class="flex flex-col gap-2">
								<Label class="flex gap-2 text-xs">Predicted Value</Label>
								<p class="h-9 text-center text-sm leading-9">{predictedVal}</p>
							</div> -->
							<!--<div class="flex flex-col gap-2">
									<div></div>
								</div>
								<div class="flex flex-col gap-2"></div>
								<div class="flex-1"></div>
								<div class="flex flex-col gap-2"></div>-->
							<br />
						</Card.Content>
					</Card.Root>
				</Tabs.Content>
				<Tabs.Content value="NL" class="h-full">
					<div class="mb-3 flex flex-row flex-wrap items-end gap-4">
						<div class="flex flex-col gap-2">
							<Label class="flex gap-2 text-xs">
								<Activity class="h-4 w-4"></Activity>
								Activation Function
							</Label>
							<Select.Root bind:selected={selectedActivation}>
								<Select.Trigger class="w-[180px]">
									<Select.Value></Select.Value>
								</Select.Trigger>
								<Select.Content>
									<Select.Item value="relu">ReLU</Select.Item>
									<Select.Item value="sigmoid">Sigmoid</Select.Item>
									<Select.Item value="tanh">Tanh</Select.Item>
									<Select.Item value="softmax">Softmax</Select.Item>
								</Select.Content>
							</Select.Root>
						</div>
						<div class="flex flex-col gap-2">
							<Label class="flex gap-2 text-xs">
								<RefreshCw class="h-4 w-4"></RefreshCw>
								Epochs
							</Label>
							<Input type="number" bind:value={epochs} placeholder="1000" min={1} class="w-24" />
						</div>
						<div class="flex flex-col gap-2">
							<Label class="flex gap-2 text-xs">
								Current Loss: {currentloss}
							</Label>
							<Button on:click={displayLoss}>
								<TrendingDown class="mr-2 h-4 w-4" /> Loss Graph
							</Button>
							<div id = "losscard" class="h-fit max-h-none w-fit max-w-none absolute translate-y-16 z-50" style="visibility:hidden">
								<Card.Root class="pt-6">
									<Card.Content>
										<Losschart pageIdx = {1} class="h-60 w-80 rounded-[0.15rem]" bind:this={losschart}/>
									</Card.Content>
								</Card.Root>
							</div>
						</div>
						<!-- <div class="flex flex-col gap-2">
							<Label class="flex gap-2 text-xs">Input</Label>
							<Input type="number" bind:value={testPred} placeholder="2" class="w-24" />
						</div>
						<div class="flex flex-col gap-2">
							<Label class="flex gap-2 text-xs">Predicted Value</Label>
							<p class="h-9 text-center text-sm leading-9">{predictedVal}</p>
						</div> -->
						<div class="flex flex-col gap-2"></div>
						<div class="flex flex-col gap-2"></div>
						<div class="flex-1"></div>
						<div class="flex flex-col gap-2">
							<Label class="flex gap-2 text-xs">Hardware</Label>
							<Tooltip.Root>
								<Tooltip.Trigger asChild>
									<div class="flex h-9 flex-row flex-nowrap items-center space-x-2">
										<Label for="hardware-backend">CPU</Label>
										<Switch id="hardware-backend" bind:checked={useGPU} />
										<Label for="hardware-backend">GPU</Label>
									</div>
								</Tooltip.Trigger>
								<Tooltip.Content class="max-w-52">
									GPU is recommended for large models but slower for small models.
								</Tooltip.Content>
							</Tooltip.Root>
						</div>
						<div class="flex flex-col gap-2">
							<Label class="flex gap-2 text-xs">Refresh</Label>
							<Button on:click={refreshModel}>
								<RefreshCw class="mr-0 h-4 w-4" />
							</Button>
						</div>
						<div class="flex flex-col gap-2">
							<Label class="flex gap-2 text-xs">Epoch: {currentEpoch}</Label>
							<Button on:click={trainModel}>
								{#if isTraining}
									<CirclePause class="mr-2 h-4 w-4"></CirclePause>
									Pause
								{:else}
									<Brain class="mr-2 h-4 w-4"></Brain>
									Train
								{/if}
							</Button>
						</div>
					</div>
					<div
						class="flex w-full flex-col gap-6 overflow-x-auto overflow-y-hidden rounded-lg border p-1 text-sm"
					>
						<div class="ml-auto mr-auto flex flex-row items-center">
							<DropdownMenu.Root>
								<DropdownMenu.Trigger>
									<Button variant="ghost" size="icon" class="h-8 w-8">
										<Plus class="h-4 w-4"></Plus>
									</Button>
								</DropdownMenu.Trigger>
								<DropdownMenu.Content>
									<DropdownMenu.Group>
										<DropdownMenu.Item on:click={() => addLayer('dense')}>Dense</DropdownMenu.Item>
										<DropdownMenu.Item
											on:click={() => {
												addLayer('conv2d');
												addLayer('maxpooling');
											}}>Convolutional</DropdownMenu.Item
										>
										<DropdownMenu.Item on:click={() => addLayer('flatten')}
											>Flatten</DropdownMenu.Item
										>
									</DropdownMenu.Group>
								</DropdownMenu.Content>
							</DropdownMenu.Root>

							<Button variant="ghost" size="icon" class="h-8 w-8" on:click={removeLayer}>
								<Minus class="h-4 w-4"></Minus>
							</Button>
							<span class="ml-2 leading-none text-muted-foreground"
								>{$model.layers.length} Layers</span
							>
						</div>

						<div class="ml-auto mr-auto flex flex-grow flex-row items-start">
							{#if tfModel}
								<Features {columnNames} {currentExample} />
								{#if $model.layers[0]?.inputShape}
									{@const weights = getWeightsBetweenLayers(tfModel, 0)}
									{#if weights}
										<ConnectionsVis
											leftLayerHeights={getNodeYPositionsInput($model.layers[0].inputShape[0])}
											rightLayerHeights={getNodeYPositions($model.layers[0])}
											{canvasWidth}
											{weights}
										/>
									{/if}
								{/if}
								{#each $model.layers as layer, i (i)}
									<svelte:component
										this={layerComponents[layer.type]}
										{layer}
										index={i}
										tfLayer={tfModel.layers[i]}
										domain={domain}
										range={range}
										columnNames={columnNames}
										currentExample={currentExample}
										{dataset}
									></svelte:component>
									{#if i < $model.layers.length - 1}
										{@const leftLayerHeights = getNodeYPositions(layer)}
										{@const rightLayerHeights = getNodeYPositions($model.layers[i + 1])}
										{@const weights = getWeightsBetweenLayers(tfModel, i + 1)}
										{#if weights}
											<ConnectionsVis
												{leftLayerHeights}
												{rightLayerHeights}
												{canvasWidth}
												{weights}
											/>
										{/if}
									{/if}
								{/each}
								<Labels
									{columnNames}
									{currentExample}
									tfLayer={tfModel.layers[tfModel.layers.length - 1]}
								/>
							{/if}
						</div>
					</div>
				</Tabs.Content>
				<Tabs.Content value="LLM" class="h-full">
					<LLM />
				</Tabs.Content>
			</Tabs.Root>
			
		</div>
	</Resizable.Pane>
	<!--</div>-->
</Resizable.PaneGroup>
