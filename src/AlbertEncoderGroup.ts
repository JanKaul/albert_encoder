import * as tf from "@tensorflow/tfjs";
import { AlbertEncoderLayer } from "./AlbertEncoderLayer";
import { nameScopeWrapper } from "./nameScope";
export class AlbertEncoderGroup extends tf.layers.Layer {
    static className = 'AlbertEncoderGroup';
    numLayersEachGroup: number;
    hiddenSize: number;
    numAttentionHeads: number;
    intermediateSize: number;
    hiddenDropoutRate: number;
    attentionDropoutRate: number;
    epsilon: number;
    initializerRange: number;
    kwargs: {};
    encoderLayers: any[];
    /* Encoder group. */
    constructor({ numLayersEachGroup = 1, hiddenSize = 768, numAttentionHeads = 12, intermediateSize = 3072, hiddenDropoutRate = 0, attentionDropoutRate = 0, epsilon = 1e-12, initializerRange = 0.02, name = "group", ...kwargs } = {}) {
        super({ name: name, ...kwargs });
        this.numLayersEachGroup = numLayersEachGroup;
        this.hiddenSize = hiddenSize;
        this.numAttentionHeads = numAttentionHeads;
        this.intermediateSize = intermediateSize;
        this.hiddenDropoutRate = hiddenDropoutRate;
        this.attentionDropoutRate = attentionDropoutRate;
        this.epsilon = epsilon;
        this.initializerRange = initializerRange;
        this.kwargs = kwargs;
        let temp = [];
        for (const i of Array(this.numLayersEachGroup).keys()) {
            temp.push(new AlbertEncoderLayer({ hiddenSize: this.hiddenSize, numAttentionHeads: this.numAttentionHeads, intermediateSize: this.intermediateSize, hiddenDropoutRate: this.hiddenDropoutRate, attentionDropoutRate: this.attentionDropoutRate, epsilon: this.epsilon, initializerRange: this.initializerRange, name: "layer_" + i.toString() }));
        };
        this.encoderLayers = temp;
    }
    call([hiddenStates, attentionMask = undefined]) {
        return tf.tidy(() => {
            let [groupHiddenStates, groupAttnWeights] = [[], []];
            for (const encoder of this.encoderLayers) {
                let attnWeights;
                [hiddenStates, attnWeights] = encoder.apply([hiddenStates, attentionMask]);
                groupHiddenStates.push(hiddenStates);
                groupAttnWeights.push(attnWeights);
            }
            return [hiddenStates, groupHiddenStates, groupAttnWeights];
        });
    }
    build([statesShape, maskShape = undefined]) {
        for (const encoder of this.encoderLayers) {
            nameScopeWrapper(encoder.name, () => { encoder.build([statesShape, maskShape]); });
            this.trainableWeights = this.trainableWeights.concat(encoder.trainableWeights);
            this.nonTrainableWeights = this.nonTrainableWeights.concat(encoder.nonTrainableWeights);
        }
        this.built = true;
    }
    getConfig() {
        const config = { numLayersEachGroup: this.numLayersEachGroup, hiddenSize: this.hiddenSize, numAttentionHeads: this.numAttentionHeads, intermediateSize: this.intermediateSize, hiddenDropoutRate: this.hiddenDropoutRate, attentionDropoutRate: this.attentionDropoutRate, epsilon: this.epsilon, initializerRange: this.initializerRange, name: this.name, ...this.kwargs };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    computeOutputShape([statesShape, maskShape = undefined]) {
        let weightsShape = this.encoderLayers[0].computeOutputShape([statesShape, maskShape])[1];
        return [statesShape, Array(this.numLayersEachGroup).fill(statesShape), Array(this.numLayersEachGroup).fill(weightsShape)];
    }
}
tf.serialization.registerClass(AlbertEncoderGroup);