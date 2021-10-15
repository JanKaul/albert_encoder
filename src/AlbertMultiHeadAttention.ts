import * as tf from "@tensorflow/tfjs";
import { nameScopeWrapper } from "./nameScope";
export class AlbertMultiHeadAttention extends tf.layers.Layer {
    static className = 'AlbertMultiHeadAttention';
    hiddenSize: number;
    numAttentionHeads: number;
    hiddenDropoutRate: number;
    attentionDropoutRate: number;
    initializerRange: number;
    epsilon: number;
    kwargs: {};
    queryWeight: any;
    keyWeight: any;
    valueWeight: any;
    dense: any;
    layerNorm: any;
    attentionDropout: any;
    outputDropout: any;
    /* Multi Head Attention. */
    constructor({ hiddenSize = 768, numAttentionHeads = 12, hiddenDropoutRate = 0.0, attentionDropoutRate = 0.0, initializerRange = 0.02, epsilon = 1e-08, name = "attention", ...kwargs } = {}) {
        super({ name: name, ...kwargs });
        this.hiddenSize = hiddenSize;
        this.numAttentionHeads = numAttentionHeads;
        this.hiddenDropoutRate = hiddenDropoutRate;
        this.attentionDropoutRate = attentionDropoutRate;
        this.initializerRange = initializerRange;
        this.epsilon = epsilon;
        this.kwargs = kwargs;
        this.queryWeight = tf.layers.dense({ units: this.hiddenSize, name: "query" });
        this.keyWeight = tf.layers.dense({ units: this.hiddenSize, name: "key" });
        this.valueWeight = tf.layers.dense({ units: this.hiddenSize, name: "value" });
        this.dense = tf.layers.dense({ units: this.hiddenSize, kernelInitializer: tf.initializers.truncatedNormal({ stddev: initializerRange }), name: "dense" });
        this.layerNorm = tf.layers.layerNormalization({ epsilon: epsilon, name: "layer_norm" });
        this.attentionDropout = tf.layers.dropout({ rate: attentionDropoutRate });
        this.outputDropout = tf.layers.dropout({ rate: hiddenDropoutRate });
    }
    _scaledDotProductAttention([query, key, value, attentionMask = undefined], { training = true } = {}) {
        let score = tf.einsum("bihj,bkhj->bihk", query, key);
        score = score.div(tf.sqrt(tf.scalar([...query.shape].pop())));
        if (attentionMask) {
            attentionMask = tf.expandDims(attentionMask, attentionMask.shape.length - 1);
            score = score.add(tf.scalar(1).sub(attentionMask).mul(tf.scalar(-1e9)));
        }
        let attnWeights = tf.softmax(score);
        attnWeights = this.attentionDropout.apply(attnWeights, { training: training });
        let context = tf.einsum("bihk,bkhl->bihl", attnWeights, value);
        return [context, attnWeights];
    }
    call([query, value, key = undefined, attentionMask = undefined], { training = true } = {}) {
        return tf.tidy(() => {
            if (!key) {
                key = value;
            }
            let originInput = query;
            let batchSize = query.shape[0];
            let valueSize = [...value.shape].pop();
            query = this.queryWeight.apply(query);
            key = this.keyWeight.apply(key);
            value = this.valueWeight.apply(value);
            query = tf.reshape(query, [batchSize, (- 1), this.numAttentionHeads, (this.hiddenSize / this.numAttentionHeads)]);
            key = tf.reshape(key, [batchSize, (- 1), this.numAttentionHeads, (this.hiddenSize / this.numAttentionHeads)]);
            value = tf.reshape(value, [batchSize, (- 1), this.numAttentionHeads, (valueSize / this.numAttentionHeads)]);
            let [context, attnWeights] = this._scaledDotProductAttention([query, key, value, attentionMask], { training: training });
            context = tf.reshape(context, [batchSize, (- 1), valueSize]);
            let output = this.dense.apply(context);
            output = this.outputDropout.apply(output, { training: training });
            output = tf.add(output, originInput);
            output = this.layerNorm.apply(output);
            return [output, attnWeights];
        });
    }
    build([queryShape, valueShape, keyShape = undefined, maskShape = undefined]) {
        nameScopeWrapper(this.queryWeight.name, () => { this.queryWeight.build(queryShape); });
        nameScopeWrapper(this.valueWeight.name, () => { this.valueWeight.build(valueShape); });
        if (keyShape) {
            nameScopeWrapper(this.keyWeight.name, () => { this.keyWeight.build(keyShape); });
        } else {
            nameScopeWrapper(this.keyWeight.name, () => { this.keyWeight.build(valueShape); });
        }
        nameScopeWrapper(this.dense.name, () => { this.dense.build([queryShape[0], queryShape[1], [...valueShape].pop()]); });
        nameScopeWrapper(this.layerNorm.name, () => { this.layerNorm.build([queryShape[0], queryShape[1], this.hiddenSize]); });
        this.trainableWeights = this.trainableWeights.concat(this.queryWeight.trainableWeights, this.valueWeight.trainableWeights, this.keyWeight.trainableWeights, this.dense.trainableWeights, this.layerNorm.trainableWeights);
        this.nonTrainableWeights = this.nonTrainableWeights.concat(this.queryWeight.nonTrainableWeights, this.valueWeight.nonTrainableWeights, this.keyWeight.nonTrainableWeights, this.dense.nonTrainableWeights, this.layerNorm.nonTrainableWeights);
        this.built = true;
    }
    getConfig() {
        const config = { hiddenSize: this.hiddenSize, numAttentionHeads: this.numAttentionHeads, hiddenDropoutRate: this.hiddenDropoutRate, attentionDropoutRate: this.attentionDropoutRate, initializerRange: this.initializerRange, epsilon: this.epsilon, ...this.kwargs };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    computeOutputShape([query_shape, value_shape, key_shape = undefined, mask_shape = undefined]) {
        return [[query_shape[0], query_shape[1], value_shape[2]], [query_shape[0], query_shape[1], value_shape[1]]]
    }
}
tf.serialization.registerClass(AlbertMultiHeadAttention);