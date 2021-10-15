import * as tf from "@tensorflow/tfjs";
import { AlbertEncoderGroup } from "./AlbertEncoderGroup";
import { nameScopeWrapper } from "./nameScope";
export class AlbertEncoder extends tf.layers.Layer {
    static className = 'AlbertEncoder';
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
    embeddingMapping: any;
    groups: any[];
    /* Encoder, stack of encoder groups. */
    constructor({ numLayers = 12, numGroups = 1, numLayersEachGroup = 1, hiddenSize = 768, numAttentionHeads = 12, intermediateSize = 3072, hiddenDropoutRate = 0, attentionDropoutRate = 0, epsilon = 1e-12, initializerRange = 0.02, name = "encoder", ...kwargs } = {}) {
        super({ name: name, ...kwargs });
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
        this.embeddingMapping = tf.layers.dense({ units: this.hiddenSize, kernelInitializer: tf.initializers.truncatedNormal({ stddev: initializerRange }), name: "embedding_mapping" });
        let temp = [];
        for (const i of Array(this.numGroups).keys()) {
            temp.push(new AlbertEncoderGroup({ numLayersEachGroup: this.numLayersEachGroup, hiddenSize: this.hiddenSize, numAttentionHeads: this.numAttentionHeads, intermediateSize: this.intermediateSize, hiddenDropoutRate: this.hiddenDropoutRate, attentionDropoutRate: this.attentionDropoutRate, epsilon: this.epsilon, initializerRange: this.initializerRange, "name": "group_" + i.toString() }));
        }
        this.groups = temp;

    }
    call([embeddings, attentionMask = undefined]) {
        return tf.tidy(() => {
            let hiddenStates = this.embeddingMapping.apply(embeddings);
            let [allHiddenStates, allAttentionWeights] = [[], []];
            for (const i of Array(this.numLayers).keys()) {
                let layersPerGroup = this.numLayers / this.numGroups;
                let groupIndex = (i / layersPerGroup >> 0);
                let groupAttnWeights, groupHiddenStates;
                [hiddenStates, groupHiddenStates, groupAttnWeights] = this.groups[groupIndex].apply([hiddenStates, attentionMask]);
                allHiddenStates = allHiddenStates.concat(groupHiddenStates);
                allAttentionWeights = allAttentionWeights.concat(groupAttnWeights);
            }
            let stackedHiddenStates = tf.stack(allHiddenStates, 0);
            stackedHiddenStates = tf.einsum('ijkl->jikl', stackedHiddenStates);
            let stackedAttentionWeights = tf.stack(allAttentionWeights, 0);
            stackedAttentionWeights = tf.einsum('ijklm->jiklm', stackedAttentionWeights);
            return [hiddenStates, stackedHiddenStates, stackedAttentionWeights];
        });
    }
    build([embedShape, maskShape = undefined]) {
        nameScopeWrapper(this.embeddingMapping.name, () => { this.embeddingMapping.build(embedShape); });
        this.trainableWeights = this.trainableWeights.concat(this.embeddingMapping.trainableWeights);
        this.nonTrainableWeights = this.nonTrainableWeights.concat(this.embeddingMapping.nonTrainableWeights);
        for (const group of this.groups) {
            nameScopeWrapper(group.name, () => { group.build([[embedShape[0], embedShape[1], this.hiddenSize], maskShape]); });
            this.trainableWeights = this.trainableWeights.concat(group.trainableWeights);
            this.nonTrainableWeights = this.nonTrainableWeights.concat(group.nonTrainableWeights);
        }
        this.built = true;
    }
    getConfig() {
        const config = { numLayers: this.numLayers, numGroups: this.numGroups, numLayersEachGroup: this.numLayersEachGroup, hiddenSize: this.hiddenSize, numAttentionHeads: this.numAttentionHeads, intermediateSize: this.intermediateSize, hiddenDropoutRate: this.hiddenDropoutRate, attentionDropoutRate: this.attentionDropoutRate, epsilon: this.epsilon, initializerRange: this.initializerRange, name: this.name, ...this.kwargs };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    computeOutputShape([embedShape, maskShape = undefined]) {
        let hiddenStateShape = [...embedShape];
        hiddenStateShape.pop();
        hiddenStateShape.push(this.hiddenSize);
        let [_, allOutputShape, allWeightShape] = this.groups[0].computeOutputShape([hiddenStateShape, maskShape]);
        allOutputShape = allOutputShape[0];
        allOutputShape.splice(1, 0, this.numLayers);
        allWeightShape = allWeightShape[0];
        allWeightShape.splice(1, 0, this.numLayers);
        return [hiddenStateShape, allOutputShape, allWeightShape];
    }
}
tf.serialization.registerClass(AlbertEncoder);