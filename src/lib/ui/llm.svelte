<script lang="ts">
    import FileInput from '$lib/components/ui/file-input/file-input.svelte';
	import Label from '$lib/components/ui/label/label.svelte';
    import * as Dialog from '$lib/components/ui/dialog';
    import { buttonVariants } from '$lib/components/ui/button';
    import Input from '$lib/components/ui/input/input.svelte';
    import * as Select from '$lib/components/ui/select';
    import Activity from 'lucide-svelte/icons/activity';
	import Brain from 'lucide-svelte/icons/brain';
	import CirclePause from 'lucide-svelte/icons/circle-pause';
	import Button from '$lib/components/ui/button/button.svelte';
    import * as tf from '@tensorflow/tfjs';
    import { writable, type Writable } from 'svelte/store';
    import { getContext, onMount, setContext } from 'svelte';
    import Losschart from '$lib/ui/losschart.svelte';
    import TrendingDown from 'lucide-svelte/icons/trending-down';
    import * as Card from '$lib/components/ui/card/index.js';
    import * as d3 from 'd3';
    import LSTMLayerVis from '$lib/ui/lstm-layer-vis.svelte'
    import Minus from 'lucide-svelte/icons/minus';
	import Plus from 'lucide-svelte/icons/plus';
    import {
        type ActivationIdentifier,
        createRNNModel,
    } from '$lib/structures';
	import DenseLayerVis from './dense-layer-vis.svelte';
	
    let textfiles: FileList;
	let text = "";
    let numUnits = 100;
    let selectedActivation = { value: 'relu' as ActivationIdentifier, label: 'ReLU' };
    let recSelectedActivation = { value: 'relu' as ActivationIdentifier, label: 'ReLU' };
    let isTraining = false;
    let currentEpoch = 0;
    let dataset: Writable<tf.data.Dataset<tf.TensorContainer>> = writable();
    let tfModel: tf.Sequential | undefined = undefined;
    let wordcount = 0;
    let losschart: Losschart;
    let currentloss = 0;
    let word_indices = new Map<string, number>();
    let indices_word = new Map<number, string>();
    let arrsetwords: string[] = [];
    let currentText = "chapter x";
    let isGenerating = false;
    let sentences: string[][] = [];
    let labels: string[][] = []

    
    const MIN_WORD_FREQUENCY = 0;
    const SEQUENCE_LEN = 7;
    const BATCH_SIZE = 32;
	$: {
		(async () => {
			if (textfiles && textfiles.length > 0) {
				textfiles.item(0)?.text().then((t) => {text=t})
			}
		})();
	}
    function encodeText(unsplit: string) {
        if (!unsplit)
            return undefined
        unsplit = unsplit.replaceAll("\r", " ").replaceAll("\n", " ");
        let seppunc = "()_-+=[]{}|~:,/".split("");
        let endpunc = ".?!;".split("");
        let punctuations = "!@#$%^&*()_+-=1234567890~`{}|[]:\";\'<>?,./".split("");
        seppunc.forEach((punc) => {unsplit = unsplit.replaceAll(punc, " ")})
        endpunc.forEach((punc) => {unsplit = unsplit.replaceAll(punc, " \n ")})
        let words = unsplit.split(" ").filter((w: string) => {return w != ''});
        let filteredwords = words.map((word: string) => {return word.split("").filter((c: string) => {return !punctuations.includes(c)}).join("").toLowerCase()}).filter((w: string) => {return w != ''});
        let word_freq = new Map<string, number>();
        filteredwords.forEach(word => {
            let val = word_freq.get(word);
            if (!val) val = 0;
            word_freq.set(word, val + 1);
        });
        
        let ignored_words = new Set<string>();
        for (let k of word_freq.keys()) {
            let val = word_freq.get(k);
            if (!val) val = 0;
            if (val < MIN_WORD_FREQUENCY)
                ignored_words.add(k);
        }
        let setwords = new Set(filteredwords);
        arrsetwords = Array.from(setwords.difference(ignored_words));
        arrsetwords.sort((n1, n2) => {
            let val1 = word_freq.get(n1);
            let val2 = word_freq.get(n2);
            if (!val1) {return 0;}
            if (!val2) {return 0;}
            if (val1 > val2) {return -1;}
            if (val1 < val2) {return 1;}
            return 0;
        });
        for (let i = 0; i < arrsetwords.length; i++) {
            word_indices.set(arrsetwords[i], i);
        }
        for (let i = 0; i < arrsetwords.length; i++) {
            indices_word.set(i, arrsetwords[i]);
        }
        
        wordcount = word_indices.size;
        for (let i = 0; i < filteredwords.length-SEQUENCE_LEN; i++) {
            let tempset = new Set(filteredwords.slice(i, i+SEQUENCE_LEN+1));
            if (tempset.intersection(ignored_words).size == 0) {
                sentences.push(filteredwords.slice(i, i+SEQUENCE_LEN));
                labels.push([filteredwords[i+SEQUENCE_LEN]]);
            }
        }
        //currentText = sentences[0].join(" ");
        return "done";
    }

    function* xgen() {
        for (let i = 0; i < sentences.length; i++) {
            yield tf.oneHot(encode(sentences[i]), word_indices.size);
        }
    }

    function* ygen() {
        for (let i = 0; i < sentences.length; i++) {
            yield tf.oneHot(encode(labels[i]), word_indices.size).as1D();
        }
    }

    function encode(arr: string[]) {
        return arr.map((word) => {
                        return word_indices.get(word);
                    }).filter((num) => {
                        return typeof num != "undefined";
                    });
    }
    function decode(arr: number[], previdx: number) {
        arr[previdx] *= 0.5
        let val = Math.random();
        let maxi = 0;
        let sum = arr.reduce((a, b)=>{return a+b}, 0);
        val *= sum;
        for (let i = 0; i < arr.length; i++) {
            val -= arr[i];
            maxi = i;
            if (val < 0) {
                break;
            }
        }
        return indices_word.get(maxi);
    }

    const generateNext = async (unsplit: string)  => {
        if (!unsplit || !tfModel)
            return undefined;
        let words = unsplit.replaceAll("\n", " \n ").split(" ").filter((w: string) => {return w != ''});
        let punctuations = "!@#$%^&*()_+-=1234567890~`{}|[]:\";\'<>?,./".split("");
        let filteredwords = words.map((word: string) => {return word.split("").filter((c: string) => {return !punctuations.includes(c)}).join("").toLowerCase()});
        if (filteredwords.length > SEQUENCE_LEN) {
            filteredwords = filteredwords.slice(-SEQUENCE_LEN);
        }
        let inputArr = encode(filteredwords);
        let zeros = [];
        if (inputArr.length < SEQUENCE_LEN) {

            for (let i = 0; i < SEQUENCE_LEN-filteredwords.length; i++) {
                zeros.push(-1);
            }
            inputArr = [...zeros, ...inputArr];
        }
        let pred = tf.oneHot(inputArr, wordcount);
        pred = pred.reshape([1, pred.shape[0], (typeof pred.shape[1] !== "undefined")?pred.shape[1]: 1]);
        let output = tfModel.predict(pred);
        
        return decode((await (output as tf.Tensor).as1D().array()) as number[], inputArr[-1]);
    }

    const generateText = async () => {
        if (!tfModel) return;
        tf.setBackend("cpu");
        if (isGenerating) {
            isGenerating = false;
            return;
        }
        
        isGenerating = true;
        while (isGenerating) {
            tf.engine().startScope();
            currentText += " " + (await generateNext(currentText));
            await delay(1000);
            tf.engine().endScope();
        }
    }
    let numEpochs = 200;

    function createModel(text: string) {
        let model;
        let val = encodeText(text);
        if (val) {
            const xs = tf.data.generator(xgen)
            const ys = tf.data.generator(ygen)
            $dataset = tf.data.zip({xs, ys}).shuffle(100).batch(BATCH_SIZE);
            model = createRNNModel(wordcount, BATCH_SIZE, SEQUENCE_LEN);
        }
        return model;
    }

    $: {
        tfModel = createModel(text);
    }

    const trainModel = async () => {
        if (!tfModel) {
            return;
        }
		if (isTraining) {
			isTraining = false;
			tfModel.stopTraining = true;
			return;
		}
        tf.setBackend('webgl');

        const data = $dataset;

        isTraining = true;
        for (let i = currentEpoch; i < numEpochs; i++) {
            tf.engine().startScope();
            const history = await tfModel.fitDataset(data, {
                epochs: 1,
                callbacks: {
                    async onEpochEnd(epoch, logs) {
                        if (!tfModel) return;
                        if (!isTraining) return;
                        currentEpoch++;
                        if (epoch % 5 === 0) tfModel = tfModel;
                        if (logs) {
                                currentloss = Math.round(logs.loss * 1000) / 1000;
                                if (losschart) {
                                    losschart.updateGraph(logs.loss);
                                }
                            }
                    }
                }
            });
            tf.engine().endScope();
        }
        tfModel = tfModel;
    }
    let losscardVisible = 0;
	function displayLoss() {
		losscardVisible = 1-losscardVisible;
		d3.select("#losscard2").style("visibility", ["hidden", "visible"][losscardVisible]);
	}
    function delay(ms: number) {
        return new Promise( resolve => setTimeout(resolve, ms) );
    }

</script>
<div class="mb-3 flex flex-row flex-wrap items-end gap-4">
    <div class="flex flex-col gap-2">
        <Label class="flex gap-2 text-xs">Choose Text File</Label>
        <Dialog.Root>
            <Dialog.Trigger class={buttonVariants({ variant: 'outline' })}
                >Upload Dataset</Dialog.Trigger
            >
            <Dialog.Content>
                <Dialog.Header>
                    <Dialog.Title>Upload Text File</Dialog.Title>
                    <Dialog.Description class="flex flex-col gap-2">
                        <p>Upload a text file to be parsed</p>
                        <div class="flex flex-col">
                            <FileInput bind:files={textfiles} />
                        </div>
                    </Dialog.Description>
                </Dialog.Header>
            </Dialog.Content>
        </Dialog.Root> 
    </div>
    <div class="flex flex-col gap-2">
        <Label class="flex gap-2 text-xs">
            LSTM Units
        </Label>
        <Input type="number" bind:value={numUnits} placeholder="100" min={1} class="w-24" />
    </div>
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
            </Select.Content>
        </Select.Root>
    </div>
    <div class="flex flex-col gap-2">
        <Label class="flex gap-2 text-xs">
            <Activity class="h-4 w-4"></Activity>
            Recurrent Activation
        </Label>
        <Select.Root bind:selected={recSelectedActivation}>
            <Select.Trigger class="w-[180px]">
                <Select.Value></Select.Value>
            </Select.Trigger>
            <Select.Content>
                <Select.Item value="relu">ReLU</Select.Item>
                <Select.Item value="sigmoid">Sigmoid</Select.Item>
                <Select.Item value="tanh">Tanh</Select.Item>
            </Select.Content>
        </Select.Root>
    </div>
    <div class="flex flex-col gap-2"></div>
    <div class="flex flex-col gap-2">
        <Label class="flex gap-2 text-xs">
            Current Loss: {currentloss}
        </Label>
        <Button on:click={displayLoss}>
            <TrendingDown class="mr-2 h-4 w-4" /> Loss Graph
        </Button>
        <div id = "losscard2" class="h-fit max-h-none w-fit max-w-none absolute translate-y-16 z-50" style="visibility:hidden">
            <Card.Root class="pt-6">
                <Card.Content>
                    <Losschart pageIdx={2} class = "h-60 w-80 rounded-[0.15rem]" bind:this={losschart}/>
                </Card.Content>
            </Card.Root>
        </div>
    </div>
    <div class="flex-1"></div>
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
{#if tfModel}
<div
class="flex w-full flex-col gap-6 overflow-x-auto overflow-y-hidden rounded-lg border p-1 text-sm"
>
    <div class="ml-auto mr-auto w-fit flex flex-row items-start">
        <LSTMLayerVis timeSteps= {SEQUENCE_LEN} units={numUnits} />
        <div class="flex flex-col items-center gap-2 rounded-lg border bg-card p-2 text-card-foreground h-fit">
            <Button variant="ghost" size="icon" class="h-6 w-6">
                <Plus class="h-4 w-4"></Plus>
            </Button>
            <Button
                variant="ghost"
                size="icon"
                class="h-6 w-6"
            >
                <Minus class="h-4 w-4"></Minus>
            </Button>
            {#each { length: 10 } as _, nodeIndex (nodeIndex)}
                <div class="relative flex h-6 w-6 items-center justify-center">
                    <div
                style={`background-color: blue;`}
                class="h-5 w-5 rounded-[0.15rem]"
                    ></div>
                </div>
            {/each}
        </div>
        <div class="flex flex-col items-start gap-2 rounded-lg py-2 text-card-foreground">
            <h5 class="mb-9 text-sm">&nbsp;</h5>
            {#each { length: 10 } as _, idx (idx)}
                <div class="flex flex-row items-center">
                    <div class="w-6 border-b border-muted-foreground"></div>
                    <div class="flex flex-row items-start justify-center rounded bg-muted px-2 py-1 font-medium">
                        <pre class="text-xs">{arrsetwords[idx]}</pre>
                    </div>
                </div>
            {/each}
        </div>
    </div>
</div>
{/if}
<div class="mb-3 flex flex-row flex-wrap items-end gap-4"> 
    <div class="flex flex-col gap-2">
        <Button on:click={generateText}>
            {#if isGenerating}
                <CirclePause class="mr-2 h-4 w-4"></CirclePause>
                Pause
            {:else}
                <Brain class="mr-2 h-4 w-4"></Brain>
                Generate
            {/if}
        </Button>
    </div>
</div>

       
<div
    class="flex w-full flex-col gap-6 overflow-x-auto overflow-y-hidden rounded-lg border p-1 text-sm"
>
<pre>{currentText}</pre>
</div>