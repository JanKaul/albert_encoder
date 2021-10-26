import * as tf from "@tensorflow/tfjs";
import { AlbertMultiHeadAttention } from './AlbertMultiHeadAttention';
import { GeluActivation } from "./GeluActivation";
import { nameScopeWrapper } from "./nameScope";
export class AlbertEncoderLayer extends tf.layers.Layer {
    static className = 'AlbertEncoderLayer';
    hiddenSize: number;
    numAttentionHeads: number;
    intermediateSize: number;
    hiddenDropoutRate: number;
    attentionDropoutRate: number;
    epsilon: number;
    initializerRange: number;
    kwargs: {};
    attention: AlbertMultiHeadAttention;
    ffn: any;
    ffnOutput: any;
    layerNorm: any;
    dropout: any;
    activation: GeluActivation;
    /* Encoder layer. */
    constructor({ hiddenSize = 768, numAttentionHeads = 12, intermediateSize = 3072, hiddenDropoutRate = 0.0, attentionDropoutRate = 0.0, epsilon = 1e-08, initializerRange = 0.02, name = "layer", ...kwargs } = {}) {
        super({ name: name, ...kwargs });
        this.hiddenSize = hiddenSize;
        this.numAttentionHeads = numAttentionHeads;
        this.intermediateSize = intermediateSize;
        this.hiddenDropoutRate = hiddenDropoutRate;
        this.attentionDropoutRate = attentionDropoutRate;
        this.epsilon = epsilon;
        this.initializerRange = initializerRange;
        this.kwargs = kwargs;
        this.attention = new AlbertMultiHeadAttention({ hiddenSize: this.hiddenSize, numAttentionHeads: this.numAttentionHeads, hiddenDropoutRate: this.hiddenDropoutRate, attentionDropoutRate: this.attentionDropoutRate, initializerRange: this.initializerRange, epsilon: this.epsilon, name: "attention" });
        this.ffn = tf.layers.dense({ units: this.intermediateSize, kernelInitializer: tf.initializers.truncatedNormal({ stddev: initializerRange }), name: "ffn" });
        this.activation = new GeluActivation();
        this.ffnOutput = tf.layers.dense({ units: this.hiddenSize, kernelInitializer: tf.initializers.truncatedNormal({ stddev: initializerRange }), name: "ffn_output" });
        this.layerNorm = tf.layers.layerNormalization({ epsilon: this.epsilon, name: "layer_norm" });
        this.dropout = tf.layers.dropout({ rate: this.hiddenDropoutRate });
    }
    call([hiddenStates, attentionMask = undefined]) {
        return tf.tidy(() => {
            if (attentionMask) {
                attentionMask = attentionMask.expandDims(1);
            }
            let state = this.attention.apply([hiddenStates, hiddenStates, hiddenStates, attentionMask]);
            let [output, weights] = [state[0], state[1]];
            let outputs = this.ffn.apply(output);
            outputs = this.activation.apply(outputs);
            outputs = this.ffnOutput.apply(outputs);
            outputs = this.dropout.apply(outputs);
            outputs = tf.add(outputs, output);
            outputs = this.layerNorm.apply(outputs);
            return [outputs, weights];
        });
    }
    build([statesShape, maskShape = undefined]) {
        nameScopeWrapper(this.attention.name, () => { this.attention.build([statesShape, statesShape, statesShape, maskShape]); });
        nameScopeWrapper(this.ffn.name, () => { this.ffn.build(statesShape); });
        nameScopeWrapper(this.ffnOutput.name, () => { this.ffnOutput.build([statesShape[0], statesShape[1], this.intermediateSize]); });
        nameScopeWrapper(this.layerNorm.name, () => { this.layerNorm.build([statesShape[0], statesShape[1], this.hiddenSize]); });
        this.trainableWeights = this.trainableWeights.concat(this.attention.trainableWeights, this.ffn.trainableWeights, this.ffnOutput.trainableWeights, this.layerNorm.trainableWeights);
        this.nonTrainableWeights = this.nonTrainableWeights.concat(this.attention.nonTrainableWeights, this.ffn.nonTrainableWeights, this.ffnOutput.nonTrainableWeights, this.layerNorm.nonTrainableWeights);
        this.built = true;
    }
    getConfig() {
        const config = { hiddenSize: this.hiddenSize, numAttentionHeads: this.numAttentionHeads, intermediateSize: this.intermediateSize, hiddenDropoutRate: this.hiddenDropoutRate, attentionDropoutRate: this.attentionDropoutRate, epsilon: this.epsilon, initializerRange: this.initializerRange, name: this.name, ...this.kwargs };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    computeOutputShape([statesShape, maskShape = undefined]) {
        return [statesShape, this.attention.computeOutputShape([statesShape, statesShape, statesShape, maskShape])[1]]
    }
}
tf.serialization.registerClass(AlbertEncoderLayer);