import * as tf from "@tensorflow/tfjs";
import { nameScopeWrapper } from "./nameScope";
export class AlbertPooler extends tf.layers.Layer {
    static className = 'AlbertPooler';
    hiddenSize: number;
    initializerRange: number;
    kwargs: {};
    dense: any;
    /* Pooler. */
    constructor({ hiddenSize = 768, initializerRange = 0.02, name = "pooler", ...kwargs } = {}) {
        super({ name: name, ...kwargs });
        this.hiddenSize = hiddenSize;
        this.initializerRange = initializerRange;
        this.kwargs = kwargs;
        this.dense = tf.layers.dense({ units: this.hiddenSize, kernelInitializer: tf.initializers.truncatedNormal({ stddev: this.initializerRange }), activation: "tanh", name: "dense" });
    }
    call([sequenceOutput]) {
        return tf.tidy(() => {
            return this.dense.apply(sequenceOutput.slice([0, 0, 0], [-1, 1, -1]).squeeze([1]));
        });
    }
    build(inputShape) {
        nameScopeWrapper(this.dense.name, () => { this.dense.build([inputShape[0], [...inputShape].pop()]); });
        this.trainableWeights = this.trainableWeights.concat(this.dense.trainableWeights);
        this.nonTrainableWeights = this.nonTrainableWeights.concat(this.dense.nonTrainableWeights);
        this.built = true;
    }
    getConfig() {
        const config = { hiddenSize: this.hiddenSize, initializerRange: this.initializerRange, name: this.name, ...this.kwargs };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    computeOutputShape(inputShape) {
        return this.dense.computeOutputShape([inputShape[0], [...inputShape].pop()]);
    }
}
tf.serialization.registerClass(AlbertPooler);