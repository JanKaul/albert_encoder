import * as tf from "@tensorflow/tfjs";
export class GeluActivation extends tf.layers.Layer {
    static className = 'GeluActivation';
    constructor({ name = "gelu" } = {}) {
        super({ name: name })
    }
    call(input) {
        let temp = tf.scalar(0.5).mul(tf.scalar(1.0).add(tf.erf(input.div(tf.sqrt(tf.scalar(2.0))))));
        return input.mul(temp);
    }

    computeOutputShape(inputShape) { return inputShape; }

    getConfig() {
        const config = { name: this.name };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
tf.serialization.registerClass(GeluActivation);