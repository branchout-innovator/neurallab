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
		updateSampledOutputs1D
	} from '$lib/structures';
	import * as tf from '@tensorflow/tfjs';
	import { onMount, setContext, SvelteComponent } from 'svelte';
	import DenseLayerVis from '$lib/ui/dense-layer-vis.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import Plus from 'lucide-svelte/icons/plus';
	import Minus from 'lucide-svelte/icons/minus';
	import Brain from 'lucide-svelte/icons/brain';
	import Activity from 'lucide-svelte/icons/activity';
	import RefreshCw from 'lucide-svelte/icons/refresh-cw';
	import { writable, type Writable } from 'svelte/store';
	import Input from '$lib/components/ui/input/input.svelte';
	import { toast } from 'svelte-sonner';
	import ConnectionsVis from '$lib/ui/connections-vis.svelte';
	import { getNodeYPositions } from '$lib/ui/connections-vis';
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

	const SAMPLE_DENSITY_2D = 20;
	const SAMPLE_DENSITY_1D = 10;
	let sample_x_domain: [number, number] = [-3, 3];
	let sample_y_domain: [number, number] = [-3, 3];

	const layerComponents: Record<string, typeof SvelteComponent> = {
		dense: DenseLayerVis as typeof SvelteComponent
	};

	let selectedActivation = { value: 'relu' as ActivationIdentifier, label: 'ReLU' };
	let epochs = 1000;

	const model = writable<SequentialModel>({
		layers: [
			{
				type: 'dense',
				units: 10,
				inputShape: [1]
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
		optimizer: 'adam'
	});
	setContext('model', model);

	const addLayer = () => {
		const lastLayer = $model.layers[$model.layers.length - 1] as DenseLayer;
		// if (lastLayer.type === 'dense')
		// 	($model.layers[$model.layers.length - 1] as DenseLayer).activation =
		// 		selectedActivation.value as ActivationIdentifier;
		$model.layers = [
			...$model.layers,
			{
				type: 'dense',
				units: 1
			} as DenseLayer
		];
		console.log($model.layers);
		updateTFModel($model);
	};

	const removeLayer = () => {
		$model.layers = [...$model.layers.slice(0, $model.layers.length - 1)];
		if ($model.layers.length > 0) {
			const lastLayer = $model.layers[$model.layers.length - 1];
			if (lastLayer.type === 'dense') {
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

	const sampleOutputs = async () => {
		if (!tfModel) return;
		if (isEqual($model.layers[0].inputShape, [1]))
			$sampledOutputs = await updateSampledOutputs1D(tfModel, SAMPLE_DENSITY_1D, sample_x_domain);
		else if (isEqual($model.layers[0].inputShape, [1]))
			$sampledOutputs = await updateSampledOutputs(
				tfModel,
				SAMPLE_DENSITY_2D,
				sample_x_domain,
				sample_y_domain
			);
	};

	const trainModel = async () => {
		if (!tfModel) return;
		// The model needs to be "big enough" to benefit from GPU acceleration
		// So with the small models in the Tensorflow playground its actually faster to use CPU
		// await tf.setBackend('webgl');
		console.log(tf.getBackend());

		const data = dataset;

		toast.loading(`Training for ${epochs} epochs...`);

		// Train the model using the data.
		currentEpoch = 0;

		try {
			await tfModel.fitDataset(data, {
				epochs: Number(epochs),
				callbacks: {
					async onEpochEnd(epoch, logs) {
						if (!tfModel) return;
						currentEpoch = epoch + 1;
						if (currentEpoch % 5 === 0) tfModel = tfModel;
						try {
							await sampleOutputs();
						} catch (e) {
							console.error('Error while sampling outputs: ', e);
						}
					}
				}
			});
			console.log(tfModel);
			console.log($model);

			// Make predictions and denormalize
			const input = tf.tensor2d([Number(testPred)], [1, 1]);
			// const normalizedInput = input.sub(data.xMin).div(data.xMax.sub(data.xMin));
			const prediction = tfModel.predict(input) as tf.Tensor; // Assert that this is a single tensor
			// const denormalizedPrediction = prediction.mul(data.yMax.sub(data.yMin)).add(data.yMin);
			prediction.print(); // Should print a value close to 4 (2^2)

			// Open the browser devtools to see the output
			toast.success(`Training complete! Try changing the input value and see the prediction.`);
			tfModel = tfModel;
		} catch (e) {
			console.error(e);
			toast.error(`Error while training: ${e}. Make sure the last layer has only 1 node.`, {
				duration: 100000
			});
		}
	};

	$: {
		for (let i = 0; i < $model.layers.length; i++) {
			const layer = $model.layers[i];
			if (layer.type === 'dense' && i < $model.layers.length - 1) {
				(layer as DenseLayer).activation = selectedActivation.value;
			}
		}
		updateTFModel($model);
	}

	function getWeightsBetweenLayers(model: tf.Sequential, layerIndex: number): tf.Tensor {
		const weights = model.layers[layerIndex].getWeights();
		return weights[0]; // Assuming the first tensor in weights array is the weight matrix
	}
	let canvasWidth = 150;

	let tfModel: tf.Sequential | undefined = undefined;
	setContext('getTfModel', () => {
		return tfModel;
	});

	const updateTFModel = async (model: SequentialModel) => {
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
		updateTFModel($model);
	}

	// $: predictedVal = tfModel?.predict(tf.tensor2d([Number(testPred)], [1, 1]));
	$: predictedVal = 0;

	let testPred = 2;

	let useGPU = false;
	$: {
		if (browser) tf.setBackend(useGPU ? 'webgl' : 'cpu');
	}

	let datasetUploadFiles: FileList;

	let dataset: tf.data.Dataset<tf.TensorContainer>;

	let csvColumnConfigs: Writable<{
		[key: string]: { isLabel: 'true' | 'false' };
	}> = writable({});

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
			} else {
				dataset = await generateDataset();
			}
		})();
	}

	let hasLabel = false;

	$: {
		if (datasetUploadFiles && datasetUploadFiles.length) {
			const config: {
				[key: string]: tf.data.ColumnConfig;
			} = {};
			hasLabel = false;
			let featureCount = 0;
			for (const column in $csvColumnConfigs) {
				const isLabel = $csvColumnConfigs[column].isLabel === 'true';
				config[column] = { isLabel };
				if (isLabel) {
					hasLabel = true;
				} else {
					featureCount++;
				}
			}
			if (hasLabel) {
				loadUploadedCsv(datasetUploadFiles[0], config).then((d) => (dataset = d));
			}
			if ($model.layers[0]) {
				$model.layers[0].inputShape = [featureCount];
				console.log('features: ', featureCount);
				updateTFModel($model);
			}
		}
	}

	function pageLeft() {
		changePage(-1);
	}
	function pageRight() {
		changePage(1);
	}
	let articletitle = [
		'sdjfnd',
		'sjokccjdj',
		'skgkoifjnm',
		'mkdjvijdmcvjijfmkijnjrjdnigjnskdnj fhdjsnd'
	];
	let pagetext = [
		'asbfhsbd',
		'sjhcbfujesndnjdjs',
		'ijgnfjvnfdnkm',
		'kbjfncmfcjfdncjfndcvfdmckvnjfdmkvjnfkmxv fdmmc'
	];
	function changePage(d: number) {
		let pageNum = Number(position);
		if ((pageNum != 0 || d != -1) && (pageNum != pagetext.length - 1 || d != 1)) {
			pageNum += d;
		}
		position = String(pageNum);
	}
	let position = '0';
	let sampledOutputs = writable<SampledOutputs<number[] | number[][]>>({});
	setContext('sampledOutputs', sampledOutputs);
	const source = `

## What are Activation Functions? (Neural Nets)

Activation functions are mathematical functions applied to the output of a neuron (like a filter). They introduce non-linearity (where input changes are not proportional to output changes) to the model, which allows it to learn and predict patterns more accurately. 
<br>
![image0](/picture0.png)
<br>
## When to use Different Activation Functions:
#### **Output Layers:**

### Binary Classification: 
The Sigmoid function is used for the output layer because it outputs a value from 0 to 1: 
σ(x) = 1/(1 + e<sup>-x</sup>)

### Multi-class Classification: 
The Softmax function is used because it converts the outputs to probabilities that sum to 100%: 
softmax(x<sub>i</sub>) = e<sup>x<sub>i</sub></sup>/∑<sub>j</sub>e<sup>x<sub>j</sub></sup>

### Regression: 
Either linear activation functions or no activation function is used. 
#### **Hidden Layers:**

### ReLU: 
Simple and effective activation function that helps to alleviate the vanishing gradient problem (derivatives of positive inputs are 1): 
ReLU(x) = max(0,x)

### Tanh: 
Similar to sigmoid but with a range of [-1, 1]. Can be more effective at training the network because of a larger gradient: 
tanh(x) = e<sup>x</sup>-e<sup>-x<sup>/e<sup>x</sup>+e<sup>-x</sup>
<br>
<br>
# Chatgpt response: 
Imagine your brain as a big, super-smart machine with lots of tiny switches inside. These switches help you decide what to do based on the information you get, like deciding whether to jump when you see a puddle or to say "hello" when you see a friend.
## Activation Functions
In a computer's brain (like in robots or apps that learn), there are also tiny switches called activation functions. These switches help the computer make decisions by turning "on" or "off" based on the information it receives. Here are some examples of how these activation functions work:
- ReLU (Rectified Linear Unit):
- Imagine a switch that stays off (at 0) if it gets a negative signal, but turns on (to the same value as the signal) when it gets a positive signal.
- It's like if you decided only to do something if it was fun (positive), and  you'd do exactly how much fun it seemed.
- Sigmoid:
- This switch smoothly turns on more and more as the signal gets bigger, but it never fully reaches 1, and never fully turns off to 0.
- Think of it like a dimmer switch for a light; as you turn it, the light gets brighter slowly, but it never gets completely dark or super bright.
- Tanh (Hyperbolic Tangent):
- This one is like the sigmoid but a bit different: it can handle both positive and negative signals, turning on for positive ones and turning off for negative ones, and it does it more smoothly.
- Imagine a balance scale: it can tip to one side for good things and to the other for bad things, showing how strong each is.
Why Are They Important?
Activation functions help the computer's brain understand and decide things better by handling information in smart ways. Just like how you use different switches or decisions based on what you're doing, computers use these activation functions to learn and make choices more accurately.
`;
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
					<Button variant="outline" class="ml-auto" size="icon" on:click={pageLeft}>&lt;</Button>
				</div>
				<div class="w-1/3">
					<DropdownMenu.Root>
						<DropdownMenu.Trigger asChild let:builder>
							<Button variant="outline" class="h-full w-full" builders={[builder]}>Pages</Button>
						</DropdownMenu.Trigger>
						<DropdownMenu.Content>
							<DropdownMenu.Label>Page Select</DropdownMenu.Label>
							<DropdownMenu.Separator />
							<DropdownMenu.RadioGroup bind:value={position}>
								{#each articletitle as title, i}
									<DropdownMenu.RadioItem value={String(i)}>{title}</DropdownMenu.RadioItem>
								{/each}
							</DropdownMenu.RadioGroup>
						</DropdownMenu.Content>
					</DropdownMenu.Root>
				</div>
				<div class="flex h-full w-1/3">
					<Button variant="outline" class="mr-auto" size="icon" on:click={pageRight}>&gt;</Button>
				</div>
			</div>
			<div class="flex w-full overflow-y-auto">
				<div class="w-full p-4">
					<h2
						class="scroll-m-20 border-b pb-2 text-center text-2xl font-semibold tracking-tight transition-colors first:mt-0"
					>
						{articletitle[Number(position)]}
					</h2>
					<span class="inline-block h-4 w-4"></span>
					<p>{pagetext[Number(position)]}</p>
					<SvelteMarkdown {source} />
				</div>
			</div>
		</div>
	</Resizable.Pane>
	<Resizable.Handle withHandle />
	<Resizable.Pane defaultSize={75}>
		<div class="flex h-full max-w-full flex-grow flex-col gap-4 overflow-x-hidden py-4">
			<!-- Controls (header) -->
			<div class="flex flex-row flex-wrap items-end gap-4">
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
					<Label class="flex gap-2 text-xs">Input</Label>
					<Input type="number" bind:value={testPred} placeholder="2" class="w-24" />
				</div>
				<div class="flex flex-col gap-2">
					<Label class="flex gap-2 text-xs">Predicted Value</Label>
					<p class="h-9 text-center text-sm leading-9">{predictedVal}</p>
				</div>
				<div class="flex flex-col gap-2">
					<div></div>
					<Dialog.Root>
						<Dialog.Trigger class={buttonVariants({ variant: 'outline' })}
							>Upload Dataset</Dialog.Trigger
						>
						<Dialog.Content>
							<Dialog.Header>
								<Dialog.Title>Upload CSV Dataset</Dialog.Title>
								<Dialog.Description class="flex flex-col gap-2">
									<p>Upload a dataset from a .csv file.</p>
									<div class="flex flex-col">
										<FileInput id="dataset-upload" class="w-32" bind:files={datasetUploadFiles} />
									</div>
									{#if Object.entries($csvColumnConfigs).length > 0}
										<Table.Root>
											<Table.Header>
												<Table.Row>
													<Table.Head class="flex-grow">Column Name</Table.Head>
												</Table.Row>
											</Table.Header>
											{#each Object.entries($csvColumnConfigs) as [column, config] (column)}
												<Table.Row>
													<Table.Cell class="font-medium">{column}</Table.Cell>
													<RadioGroup.Root bind:value={$csvColumnConfigs[column].isLabel} asChild>
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
								</Dialog.Description>
							</Dialog.Header>
						</Dialog.Content>
					</Dialog.Root>
				</div>
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
					<Label class="flex gap-2 text-xs">Epoch: {currentEpoch}</Label>
					<Button on:click={trainModel}>
						<Brain class="mr-2 h-4 w-4"></Brain>
						Train
					</Button>
				</div>
			</div>
			<div
				class="flex h-full w-full flex-col gap-6 overflow-x-scroll rounded-lg border p-6 text-sm"
			>
				<div class="ml-auto mr-auto flex flex-row items-center">
					<Button variant="ghost" size="icon" class="h-8 w-8" on:click={addLayer}>
						<Plus class="h-4 w-4"></Plus>
					</Button>
					<Button variant="ghost" size="icon" class="h-8 w-8" on:click={removeLayer}>
						<Minus class="h-4 w-4"></Minus>
					</Button>
					<span class="ml-2 leading-none text-muted-foreground">{$model.layers.length} Layers</span>
				</div>

				<div class="ml-auto mr-auto flex flex-grow flex-row items-start">
					{#if tfModel}
						{#each $model.layers as layer, i (i)}
							<svelte:component
								this={layerComponents[layer.type]}
								{layer}
								index={i}
								tfLayer={tfModel.layers[i]}
							></svelte:component>
							{#if i < $model.layers.length - 1}
								{@const leftLayerHeights = getNodeYPositions(layer)}
								{@const rightLayerHeights = getNodeYPositions($model.layers[i + 1])}
								{@const weights = getWeightsBetweenLayers(tfModel, i + 1)}
								<ConnectionsVis {leftLayerHeights} {rightLayerHeights} {canvasWidth} {weights} />
							{/if}
						{/each}
					{/if}
				</div>
			</div>
		</div>
	</Resizable.Pane>
	<!--</div>-->
</Resizable.PaneGroup>
