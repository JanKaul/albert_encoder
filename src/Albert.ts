import * as tf from "@tensorflow/tfjs";
import { AlbertEmbedding } from "./AlbertEmbedding";
import { AlbertEncoder } from "./AlbertEncoder";
import { AlbertPooler } from "./AlbertPooler";
import { nameScopeWrapper } from "./nameScope";
export class Albert extends tf.layers.Layer {
    static className = 'Albert';
    vocabSize: number;
    maxPositions: number;
    embeddingSize: number;
    typeVocabSize: number;
    numLayers: number;
    numGroups: number;
    numLayersEachGroup: number;
    hiddenSize: number;
    numAttentionHeads: number;
    intermediateSize: number;
    hiddenDropoutRate: number;
    attentionDropoutRate: number;
    epsilon: number;
    initializerRange: number;
    kwargs: {};
    embedding: AlbertEmbedding;
    encoder: AlbertEncoder;
    pooler: AlbertPooler;
    returnStates: boolean;
    returnAttentionWeights: boolean;
    /* Albert model. */
    constructor({ vocabSize = 30000, maxPositions = 512, embeddingSize = 128, typeVocabSize = 2, numLayers = 12, numGroups = 1, numLayersEachGroup = 1, hiddenSize = 768, numAttentionHeads = 12, intermediateSize = 3072, hiddenDropoutRate = 0, attentionDropoutRate = 0, epsilon = 1e-12, initializerRange = 0.02, returnStates = false, returnAttentionWeights = false, name = "albert", ...kwargs } = {}) {
        super({ name: name, ...kwargs });
        this.vocabSize = vocabSize;
        this.maxPositions = maxPositions;
        this.embeddingSize = embeddingSize;
        this.typeVocabSize = typeVocabSize;
        this.numLayers = numLayers;
        this.numGroups = numGroups;
        this.numLayersEachGroup = numLayersEachGroup;
        this.hiddenSize = hiddenSize;
        this.numAttentionHeads = numAttentionHeads;
        this.intermediateSize = intermediateSize;
        this.hiddenDropoutRate = hiddenDropoutRate;
        this.attentionDropoutRate = attentionDropoutRate;
        this.epsilon = epsilon;
        this.initializerRange = initializerRange;
        this.kwargs = kwargs;
        this.embedding = new AlbertEmbedding({ vocabSize: this.vocabSize, maxPositions: this.maxPositions, embeddingSize: this.embeddingSize, typeVocabSize: this.typeVocabSize, hiddenDropoutRate: this.hiddenDropoutRate, epsilon: this.epsilon, initializerRange: this.initializerRange, name: "embeddings" });
        this.encoder = new AlbertEncoder({ numLayers: this.numLayers, numGroups: this.numGroups, numLayersEachGroup: this.numLayersEachGroup, hiddenSize: this.hiddenSize, numAttentionHeads: this.numAttentionHeads, intermediateSize: this.intermediateSize, hiddenDropoutRate: this.hiddenDropoutRate, attentionDropoutRate: this.attentionDropoutRate, epsilon: this.epsilon, initializerRange: this.initializerRange, name: "encoder" });
        this.pooler = new AlbertPooler({ hiddenSize: this.hiddenSize, initializerRange: this.initializerRange, name: "pooler" });
        this.returnStates = returnStates;
        this.returnAttentionWeights = returnAttentionWeights;
    }
    call([inputIds, segmentIds = undefined, attentionMask = undefined]) {
        return tf.tidy(() => {
            let embed = this.embedding.apply([inputIds, segmentIds]);
            let state = this.encoder.apply([embed, attentionMask]);
            let pooledOutput = this.pooler.apply([state[0]]);
            let outputs = [state[0], pooledOutput];
            if (this.returnStates) {
                outputs.push(state[1]);
            }
            if (this.returnAttentionWeights) {
                outputs.push(state[2]);
            }
            return outputs;
        });
    }
    build([idShape, segmentShape = undefined, maskShape = undefined]) {
        nameScopeWrapper(this.embedding.name, () => { this.embedding.build([idShape, segmentShape]); });
        nameScopeWrapper(this.encoder.name, () => { this.encoder.build([[idShape[0], idShape[1], this.embeddingSize], maskShape]); });
        nameScopeWrapper(this.pooler.name, () => { this.pooler.build([idShape[0], idShape[1], this.hiddenSize]); });
        this.trainableWeights = this.trainableWeights.concat(this.embedding.trainableWeights, this.encoder.trainableWeights, this.pooler.trainableWeights);
        this.nonTrainableWeights = this.nonTrainableWeights.concat(this.embedding.nonTrainableWeights, this.encoder.nonTrainableWeights, this.pooler.nonTrainableWeights);
        this.built = true;
    }
    getConfig() {
        const config = { vocabSize: this.vocabSize, maxPositions: this.maxPositions, embeddingSize: this.embeddingSize, typeVocabSize: this.typeVocabSize, numLayers: this.numLayers, numGroups: this.numGroups, numLayersEachGroup: this.numLayersEachGroup, hiddenSize: this.hiddenSize, numAttentionHeads: this.numAttentionHeads, intermediateSize: this.intermediateSize, hiddenDropoutRate: this.hiddenDropoutRate, attentionDropoutRate: this.attentionDropoutRate, epsilon: this.epsilon, initializerRange: this.initializerRange, returnStates: this.returnStates, returnAttentionWeights: this.returnAttentionWeights, name: this.name, ...this.kwargs };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    computeOutputShape([idShape, segmentShape = undefined, maskShape = undefined]) {
        let embedShape = this.embedding.computeOutputShape([idShape, segmentShape]);
        let attentionShape = this.encoder.computeOutputShape([embedShape, maskShape]);
        let pooledShape = this.pooler.computeOutputShape(attentionShape[0]);
        let outputShape = [attentionShape[0], pooledShape];
        if (this.returnStates) {
            outputShape.push(attentionShape[1]);
        }
        if (this.returnAttentionWeights) {
            outputShape.push(attentionShape[2]);
        }
        return outputShape
    }
}
tf.serialization.registerClass(Albert);